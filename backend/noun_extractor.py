"""
Gemini 기반 핵심 개념 추출기 (v2 — kiwipiepy 완전 제거).

기존 문제:
  kiwipiepy 형태소 분석 → "도널드" + "트럼프" 따로 추출
  → STOPWORDS 필터 → Gemini 단어 목록 판별 (맥락 없음)
  → 의미 없는 공동출현 그래프 ("도널드" ↔ "트럼프" 엣지 최상위 랭크)

신규 방식:
  Gemini가 기사를 직접 읽고 분석에 의미 있는 핵심 개념 5~15개 추출
  → "트럼프 관세", "연준 금리 동결", "반도체 공급망 재편" 같은
     맥락이 담긴 복합 개념 노드 생성
  → 공동출현 그래프가 실제 경제 지식을 반영
"""

import json
from typing import List, Dict

from database import get_db

from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── 프롬프트 ────────────────────────────────────────────────────
CONCEPT_PROMPT = """다음 뉴스 기사에서 경제·지정학 분석에 의미 있는 핵심 개념을 추출해주세요.

기사:
{text}

추출 기준:
- 인물·기관: "트럼프 행정부", "연준(Fed)", "OPEC+" 처럼 맥락 포함 형태 권장
- 정책·이벤트: "트럼프 관세", "FOMC 금리 동결", "홍해 봉쇄" 처럼 구체적으로
- 경제 지표: 기사에서 실제 언급된 CPI, 기준금리, 환율, 유가 등
- 복합 명사 허용: 2~4어절의 구체적 표현 (예: "중국 희토류 수출 제한", "반도체 공급망")
- 지명은 단독 금지, 반드시 사건·맥락과 결합 (예: "대만" X → "대만 해협 긴장" O)

금지:
- "경제", "시장", "상황", "문제", "관계", "영향", "전망" 같은 지나치게 일반적인 단어
- 기사에서 핵심이 아닌 주변적 언급 (기자 이름, 사진 설명 등)
- 단순 고유명사 분절 ("도널드", "트럼프" 따로 — "트럼프" 하나로)

반환 수: 5~15개 (기사 내용이 풍부하면 많이, 단순하면 적게)

JSON 배열로만 응답 (설명, 마크다운 없이):
["개념1", "개념2", ...]"""


# ══════════════════════════════════════════════════════════════
#  단일 기사 개념 추출
# ══════════════════════════════════════════════════════════════

def extract_concepts_from_article(title: str, body: str) -> List[str]:
    """
    Gemini로 기사 한 건에서 핵심 개념 추출.
    반환: ["트럼프 관세", "반도체 공급망", ...]
    """
    if not GEMINI_API_KEY:
        return []

    text = f"{title}\n\n{body[:3000]}"
    prompt = CONCEPT_PROMPT.format(text=text)

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(
            model="gemini-2.0-flash",   # 빠르고 저렴한 모델로 대량 처리
            contents=prompt,
        )
        raw = resp.text.strip()

        # 마크다운 코드블록 제거
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()

        concepts = json.loads(raw)

        if not isinstance(concepts, list):
            return []

        # 검증: 문자열이고 2자 이상, 30자 이하
        valid = [
            c.strip() for c in concepts
            if isinstance(c, str)
            and 2 <= len(c.strip()) <= 30
        ]
        return valid[:20]   # 최대 20개

    except Exception as e:
        print(f"[Concept] 추출 오류: {e}")
        return []


# ══════════════════════════════════════════════════════════════
#  배치 처리
# ══════════════════════════════════════════════════════════════

def process_articles(batch_size: int = 50, use_gemini_filter: bool = True) -> dict:
    """
    DB의 미처리 기사들에서 Gemini로 핵심 개념 추출 후 article_nouns에 저장.
    use_gemini_filter 파라미터는 하위 호환성을 위해 유지 (항상 Gemini 사용).
    """
    processed   = 0
    total_nouns = 0

    with get_db() as conn:
        # 아직 개념 추출 안 된 기사 조회
        # (article_nouns에 전혀 없는 기사 — __processed__ 마커 포함 검사)
        articles = conn.execute("""
            SELECT a.id, a.title, a.body
            FROM articles a
            LEFT JOIN article_nouns an ON a.id = an.article_id
            WHERE an.id IS NULL
            ORDER BY a.published_at DESC
            LIMIT ?
        """, (batch_size,)).fetchall()

    if not articles:
        return {"processed_articles": 0, "total_nouns": 0, "message": "처리할 기사 없음"}

    for article in articles:
        art_id = article["id"]
        title  = article["title"] or ""
        body   = article["body"]  or ""

        concepts = extract_concepts_from_article(title, body)

        with get_db() as conn:
            if concepts:
                for concept in concepts:
                    # 제목에 포함된 개념은 'title' position으로 저장 (가중치 부여용)
                    position = "title" if concept in title else "body"
                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO article_nouns
                               (article_id, noun, position) VALUES (?, ?, ?)""",
                            (art_id, concept, position),
                        )
                    except Exception as e:
                        print(f"[Concept] DB 저장 오류: {e}")
                total_nouns += len(concepts)
            else:
                # 개념이 없어도 처리됨 마킹 (재처리 방지)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO article_nouns
                           (article_id, noun, position) VALUES (?, ?, ?)""",
                        (art_id, "__processed__", "body"),
                    )
                except Exception:
                    pass

        processed += 1
        if processed % 10 == 0:
            print(f"  [{processed}/{len(articles)}] 개념 추출 진행 중...")

    print(f"개념 추출 완료: {processed}개 기사, {total_nouns}개 개념")
    return {"processed_articles": processed, "total_nouns": total_nouns}


# ══════════════════════════════════════════════════════════════
#  초기화 (기존 쓰레기 데이터 클리어)
# ══════════════════════════════════════════════════════════════

def reset_concepts() -> dict:
    """
    기존 article_nouns / nodes / edges 전부 삭제.
    kiwipiepy 기반으로 생성된 저품질 데이터를 날리고 새로 시작.
    """
    with get_db() as conn:
        noun_count = conn.execute("SELECT COUNT(*) as c FROM article_nouns").fetchone()["c"]
        node_count = conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
        edge_count = conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]

        conn.execute("DELETE FROM article_nouns")
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM nodes")

    print(f"초기화 완료: article_nouns {noun_count}개, nodes {node_count}개, edges {edge_count}개 삭제")
    return {
        "deleted_article_nouns": noun_count,
        "deleted_nodes":         node_count,
        "deleted_edges":         edge_count,
        "message": "초기화 완료. /api/pipeline/extract 로 재처리하세요.",
    }


# ══════════════════════════════════════════════════════════════
#  직접 실행 테스트
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_title = "트럼프, 반도체 관세 25% 추가 부과 검토…삼성·SK하이닉스 타격"
    test_body  = """
    도널드 트럼프 미국 대통령이 한국·대만산 반도체에 25% 추가 관세를 부과하는 방안을
    검토 중이라고 월스트리트저널이 보도했다. 이에 따라 삼성전자와 SK하이닉스 주가가
    급락했으며, 코스피 지수도 하락세를 보였다. TSMC 등 대만 반도체 기업들도 영향권에
    들어갈 것으로 예상되며, 공급망 재편 논의가 가속화될 전망이다.
    미 연준(Fed)은 이와 별개로 기준금리를 동결할 것으로 보인다.
    """

    print("=== 기존 방식 결과 (예시) ===")
    print("도널드, 트럼프, 대통령, 관세, 검토, 한국, 대만, 반도체, 부과, ...")
    print()
    print("=== 새 Gemini 방식 결과 ===")
    concepts = extract_concepts_from_article(test_title, test_body)
    print(concepts)

    from database import init_db
    init_db()
    result = process_articles(batch_size=3)
    print(result)
