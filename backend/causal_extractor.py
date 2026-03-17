"""
Gemini 기반 인과 체인 추출기 (v2 — 체인 중심 설계).

핵심 변경:
  - 기존: (cause, effect) 쌍 단위 추출 → 최대 5개 제한, 맥락 손실
  - 신규: 카테고리별 전체 체인 추출 → 제한 없음, 체인 원본 보존

목표: "전쟁이 일어나면 무엇이 영향받나?" 형태의 질의에 직접 대응하는
      방향 있는 인과 지식 그래프 축적.

DB 구조:
  causal_chains: 체인 전체 원본 저장 (category, chain_text, confidence, extractor_version)
  causal_edges:  체인을 (cause → effect) 쌍으로 분해해 그래프 탐색에 활용

버전 관리:
  USE_THINKING = True  → v1 설정 (thinking 활성화, 기존 프롬프트)
  USE_THINKING = False → v2 설정 (thinking_budget=0, 강화 프롬프트)
  extractor_version 컬럼으로 DB 내 v1/v2 구분 가능
  롤백: results_v1/ 폴더의 macro_graph_v1_snapshot.db 으로 복원
"""

import os
import json
import time
from typing import List, Dict, Optional
from collections import deque

from database import get_db

from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ══════════════════════════════════════════════════════════════
#  버전 플래그 — 여기만 바꾸면 v1/v2 전환
# ══════════════════════════════════════════════════════════════
USE_THINKING = False   # True = v1(thinking 활성화), False = v2(thinking 비활성화+강화 프롬프트)

# ── 학습 데이터 저장 경로 ────────────────────────────────────────
_BACKEND_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR  = os.path.join(_BACKEND_DIR, "training_data")
TRAINING_JSONL     = os.path.join(TRAINING_DATA_DIR, "causal_chains.jsonl")   # ChatML 포맷
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

# ── 결과물 버전 디렉토리 ─────────────────────────────────────────
RESULTS_V1_DIR = os.path.join(_BACKEND_DIR, "results_v1")
RESULTS_V2_DIR = os.path.join(_BACKEND_DIR, "results_v2")
os.makedirs(RESULTS_V1_DIR, exist_ok=True)
os.makedirs(RESULTS_V2_DIR, exist_ok=True)


def _save_training_example(article_text: str, raw_output: str, system_prompt: str,
                           version: str = "v1") -> None:
    """
    Gemini 입출력 쌍을 ChatML 포맷 JSONL로 저장.

    포맷 (Llama 3 / Mistral / Qwen 등 범용):
    {
      "messages": [
        {"role": "system",    "content": "<시스템 프롬프트>"},
        {"role": "user",      "content": "<기사 텍스트>"},
        {"role": "assistant", "content": "<JSON 응답>"}
      ],
      "extractor_version": "v1" | "v2"
    }

    raw_output: Gemini가 실제로 반환한 텍스트 (마크다운 포함 원본)
                → 로컬 모델이 동일한 출력 형식을 학습
    version: 추출기 버전 태그 ("v1" 또는 "v2")
    """
    example = {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": article_text},
            {"role": "assistant", "content": raw_output},
        ],
        "extractor_version": version,
    }
    try:
        with open(TRAINING_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [학습 데이터 저장 오류] {e}")

# ── 허용 카테고리 (추출 시 검증용) ─────────────────────────────
VALID_CATEGORIES = {
    "실물경제", "지정학", "에너지시장", "공급망",
    "금융시장", "통화패권", "기술산업", "안보", "식량", "보건",
}

# ── 프롬프트 (v1 — 기존 보존, USE_THINKING=True 시 사용) ────────
# [v1 원본 프롬프트 — 수정하지 말 것]
CHAIN_PROMPT_V1 = """다음 뉴스 기사를 읽고, 인과 체인을 최대한 많이 추출해주세요.

목표: "전쟁이 일어나면 어떻게 되지?", "유가가 오르면 무엇이 영향받나?" 같은
     질문에 답할 수 있는 인과 지식 그래프를 구축하는 것.

기사:
{text}

추출 규칙:
- 체인은 A → B → C → D 형태의 연속 인과 흐름
- 노드는 2~10자의 구체적인 명사구 (예: "호르무즈 봉쇄", "유가 급등", "CPI 상승")
- "상승", "하락" 같은 방향어 단독 금지 → 반드시 주체와 결합 (예: "유가 상승" O, "상승" X)
- 체인 길이: 최소 3단계, 최대 8단계
- 카테고리: 체인이 속하는 영역
  (실물경제 / 지정학 / 에너지시장 / 공급망 / 금융시장 / 통화패권 / 기술산업 / 안보 / 식량 / 보건 중 선택)
- confidence: 1=기사에서 간접 암시, 2=기사에서 명시적 언급, 3=기사의 핵심 인과
- 기사에 없는 내용 추론 절대 금지
- 가능한 한 많이 추출 (수 제한 없음)

JSON 배열로만 응답 (설명 없이, 마크다운 없이):
[
  {{"category": "카테고리", "chain": ["노드1", "노드2", "노드3", ...], "confidence": 1~3}},
  ...
]"""

# ── 프롬프트 (v2 — thinking 비활성화 대응 강화 버전, USE_THINKING=False 시 사용) ──
# 변경 포인트:
#   4-1. 출력 포맷 명시 강화 (JSON 배열만, 다른 텍스트 절대 금지)
#   4-2. 인과관계 추출 규칙 명시화 (A→B→C 직접 인과만, 상관관계 제외)
#   4-3. Chain-of-thought 유도 문장 추가
CHAIN_PROMPT_V2 = """당신은 뉴스 기사에서 경제·지정학 인과 체인을 추출하는 전문 분석기입니다.

목표: "전쟁이 일어나면 어떻게 되지?", "유가가 오르면 무엇이 영향받나?" 같은
     질문에 답할 수 있는 인과 지식 그래프를 구축하는 것.

기사:
{text}

━━━ 인과관계 추출 규칙 ━━━
[필수 형태]
- 반드시 "A → B → C" 형태의 직접 인과 흐름만 추출
- A가 원인이 되어 B가 발생하고, B가 원인이 되어 C가 발생하는 직접 인과만 허용
- 단순 상관관계, 시간적 나열, 배경 설명은 절대 제외

[노드 작성 규칙]
- 각 노드는 15자 이내 명사구로 작성 (예: "호르무즈 봉쇄", "유가 급등", "CPI 상승")
- "상승", "하락" 같은 방향어 단독 금지 → 반드시 주체와 결합 (예: "유가 상승" O, "상승" X)
- 기사에 없는 내용 추론 절대 금지

[체인 형식]
- 체인 길이: 최소 3개 노드, 최대 8개 노드
- 동일 기사에서 최소 3개, 최대 15개 체인 추출
- 카테고리: 실물경제 / 지정학 / 에너지시장 / 공급망 / 금융시장 / 통화패권 / 기술산업 / 안보 / 식량 / 보건 중 하나
- confidence: 1=기사에서 간접 암시, 2=기사에서 명시적 언급, 3=기사의 핵심 인과

━━━ 출력 형식 (엄수) ━━━
- 응답은 반드시 JSON 배열만 출력 — 앞뒤 설명, 마크다운 코드블록(```) 절대 금지
- 최상위는 반드시 [ ] 배열
- 각 객체는 반드시 "category", "chain", "confidence" 키 3개만 포함
- "chain" 값은 반드시 3개 이상 노드를 가진 문자열 리스트

출력 예시:
[
  {{"category": "에너지시장", "chain": ["호르무즈 봉쇄", "원유 공급 차질", "유가 급등", "CPI 상승"], "confidence": 3}},
  {{"category": "금융시장", "chain": ["유가 급등", "인플레 압력", "금리 인상", "달러 강세"], "confidence": 2}}
]

━━━ 추출 전 사고 순서 ━━━
추출 전에 기사의 핵심 사건을 먼저 파악하고,
각 사건 간 직접적 인과관계만 선별한 뒤 JSON으로 출력하라."""

# 현재 사용할 프롬프트 (USE_THINKING 플래그로 자동 선택됨)
CHAIN_PROMPT = CHAIN_PROMPT_V1 if USE_THINKING else CHAIN_PROMPT_V2


# ══════════════════════════════════════════════════════════════
#  체인 추출
# ══════════════════════════════════════════════════════════════

def extract_chains_from_article(article_id: int, title: str, body: str,
                                max_retries: int = 4) -> List[Dict]:
    """
    기사 한 건에서 Gemini로 인과 체인 추출.
    Rate limit(429) 발생 시 지수 백오프로 최대 max_retries회 재시도.

    USE_THINKING 플래그에 따라 v1/v2 설정 자동 분기:
      - USE_THINKING=True  → v1: thinking 활성화, CHAIN_PROMPT_V1
      - USE_THINKING=False → v2: thinking_budget=0, CHAIN_PROMPT_V2

    반환:
        [{"category": "실물경제", "chain": ["A","B","C"], "confidence": 3,
          "article_id": 1, "extractor_version": "v1"/"v2"}, ...]
    """
    if not GEMINI_API_KEY:
        return []

    # 버전 결정
    extractor_version = "v1" if USE_THINKING else "v2"
    current_prompt    = CHAIN_PROMPT_V1 if USE_THINKING else CHAIN_PROMPT_V2

    text   = f"{title}\n\n{body[:4000]}"
    prompt = current_prompt.format(text=text)

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    # ── generation_config 분기 ─────────────────────────────────
    # [v1 설정 — 원본 보존]
    # gen_config_v1 = {}  # thinking 기본값 사용 (활성화)
    #
    # [v2 설정 — thinking 비활성화]
    # thinking_budget=0 으로 설정하면 모델이 즉시 응답 생성
    # 모델은 gemini-2.5-flash 유지 (flash-lite로 변경 금지)
    if USE_THINKING:
        # v1: thinking 기본값 (활성화) — 기존 동작 그대로
        gen_config = {}
    else:
        # v2: thinking 완전 비활성화
        gen_config = {
            "thinking_config": {
                "thinking_budget": 0,
            }
        }

    for attempt in range(1, max_retries + 1):
        try:
            # v1/v2 분기 호출
            if USE_THINKING:
                # [v1 원본 호출 방식 — 보존]
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
            else:
                # [v2 호출 — thinking_budget=0 적용]
                from google.genai import types as genai_types
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        thinking_config=genai_types.ThinkingConfig(
                            thinking_budget=0,
                        ),
                    ),
                )

            raw = resp.text.strip()

            # 토큰 사용량 출력 (usage_metadata)
            usage = getattr(resp, "usage_metadata", None)
            if usage:
                print(f"    [토큰] in={getattr(usage,'prompt_token_count','-')} "
                      f"out={getattr(usage,'candidates_token_count','-')} "
                      f"think={getattr(usage,'thoughts_token_count','-')} "
                      f"total={getattr(usage,'total_token_count','-')} "
                      f"[{extractor_version}]")

            # 마크다운 코드블록 제거 (v2에서는 발생 안 해야 하지만 방어 처리)
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.split("```")[0].strip()

            chains = json.loads(raw)
            valid  = []
            for c in chains:
                if not isinstance(c, dict):
                    continue
                chain_nodes = c.get("chain", [])
                raw_cat     = str(c.get("category", "기타")).strip()
                # 허용 목록 외 카테고리는 가장 유사한 것으로 매핑, 없으면 "기타"
                category    = raw_cat if raw_cat in VALID_CATEGORIES else "기타"
                confidence  = int(c.get("confidence", 1))

                if len(chain_nodes) < 3:
                    continue
                cleaned = [n.strip()[:15] for n in chain_nodes if len(n.strip()) >= 2]
                if len(cleaned) < 3:
                    continue
                if len(set(cleaned)) != len(cleaned):
                    continue

                valid.append({
                    "category":          category,
                    "chain":             cleaned,
                    "confidence":        min(max(confidence, 1), 3),
                    "article_id":        article_id,
                    "extractor_version": extractor_version,
                })

            # ── 성공한 경우에만 학습 데이터 저장 ──
            # 실패·빈 응답은 노이즈이므로 저장 안 함
            if valid:
                _save_training_example(
                    article_text=text,
                    raw_output=raw,
                    system_prompt=current_prompt.split("기사:")[0].strip(),
                    version=extractor_version,
                )

            return valid

        except Exception as e:
            err_str = str(e).lower()

            # Rate limit → 대기 후 재시도
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                wait = 60 * attempt   # 1분 → 2분 → 3분 → 4분
                print(f"  [Rate Limit] {attempt}/{max_retries}회 재시도 예정 — {wait}초 대기...")
                print(f"   에러 내용: {e}")
                time.sleep(wait)
                continue

            # JSON 파싱 실패 → 재시도 없이 포기 (Gemini 응답 형식 문제)
            if "json" in err_str or "expecting" in err_str:
                print(f"  [JSON 오류] article_id={article_id}: {e}")
                return []

            # 그 외 오류 → 재시도 없이 포기
            print(f"  [Chain 오류] article_id={article_id} (시도 {attempt}): {e}")
            return []

    print(f"  [포기] article_id={article_id} — Rate Limit 재시도 횟수 초과")
    return []


# ══════════════════════════════════════════════════════════════
#  체인 → 엣지 변환
# ══════════════════════════════════════════════════════════════

def chains_to_edges(chains: List[Dict]) -> List[Dict]:
    """
    체인 리스트를 (cause → effect) 엣지 목록으로 분해.
    체인 내 연속 노드 쌍이 모두 엣지가 되며, chain_text와 extractor_version을 태그로 유지.
    """
    edges = []
    for c in chains:
        nodes      = c["chain"]
        chain_text = " → ".join(nodes)
        ev         = c.get("extractor_version", "v1")
        for i in range(len(nodes) - 1):
            edges.append({
                "cause":             nodes[i],
                "effect":            nodes[i + 1],
                "category":          c["category"],
                "relation":          f"{nodes[i]}→{nodes[i+1]}"[:20],
                "strength":          c["confidence"],
                "article_id":        c["article_id"],
                "chain_text":        chain_text,
                "extractor_version": ev,
            })
    return edges


# ══════════════════════════════════════════════════════════════
#  배치 처리 (미처리 기사 → DB 저장)
# ══════════════════════════════════════════════════════════════

def process_articles_for_chains(batch_size: int = 30, rate_limit_sec: float = 7.0) -> Dict:
    """
    causal_chains 테이블에 아직 등록 안 된 기사들을 처리.
    체인, 엣지, article_nouns(체인 노드)를 동시에 저장.

    rate_limit_sec: 호출 간 대기 시간 (무료 10 RPM → 7초 권장)
    noun_extractor 별도 실행 불필요 — 체인 노드가 곧 고품질 개념 노드.
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY 없음"}

    with get_db() as conn:
        articles = conn.execute("""
            SELECT a.id, a.title, a.body
            FROM articles a
            WHERE a.id NOT IN (
                SELECT DISTINCT article_id FROM causal_chains
                WHERE article_id IS NOT NULL
            )
            AND a.body IS NOT NULL AND length(a.body) > 100
            ORDER BY a.published_at DESC
            LIMIT ?
        """, (batch_size,)).fetchall()

    if not articles:
        return {"processed": 0, "chains_saved": 0, "edges_saved": 0,
                "message": "처리할 기사 없음"}

    total_chains     = 0
    total_edges      = 0
    new_edges        = 0   # 실제로 새로 삽입된 엣지 (기존에 없던 연결)
    reinforced_edges = 0   # 이미 있던 엣지에 증거 추가 (INSERT OR IGNORE로 스킵된 것)
    total_nouns      = 0
    new_nodes        = 0   # 처음 등장하는 노드
    processed        = 0
    skipped          = 0

    for i, art in enumerate(articles):
        try:
            chains = extract_chains_from_article(
                art["id"], art["title"] or "", art["body"] or ""
            )
        except Exception as e:
            # extract 함수 밖에서 터지는 예상치 못한 예외도 잡아서 계속 진행
            print(f"  [예외] article_id={art['id']} 건너뜀: {e}")
            processed += 1
            if i < len(articles) - 1:
                time.sleep(rate_limit_sec)
            continue

        if chains:
            edges = chains_to_edges(chains)

            # 모든 체인에서 고유 노드 수집 → article_nouns에 저장
            all_nodes = set()
            for c in chains:
                for node in c["chain"]:
                    all_nodes.add(node)

            with get_db() as conn:
                # ── 체인 원본 저장 ──
                for c in chains:
                    chain_text = " → ".join(c["chain"])
                    ev = c.get("extractor_version", "v1")
                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO causal_chains
                               (article_id, category, chain_text, confidence, extractor_version)
                               VALUES (?, ?, ?, ?, ?)""",
                            (c["article_id"], c["category"],
                             chain_text, c["confidence"], ev),
                        )
                        total_chains += 1
                    except Exception as e:
                        print(f"[Chain] chains 저장 오류: {e}")

                # ── 엣지 저장 (신규/강화 분리 카운트) ──
                for e in edges:
                    ev = e.get("extractor_version", "v1")
                    try:
                        before = conn.total_changes
                        conn.execute(
                            """INSERT OR IGNORE INTO causal_edges
                               (cause, effect, relation, strength,
                                article_id, category, chain_text, extractor_version)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (e["cause"], e["effect"], e["relation"],
                             e["strength"], e["article_id"],
                             e["category"], e["chain_text"], ev),
                        )
                        total_edges += 1
                        if conn.total_changes > before:
                            new_edges += 1        # 실제로 삽입됨 = 새 연결
                        else:
                            reinforced_edges += 1  # 이미 있던 연결 = 증거 누적
                    except Exception as e_err:
                        print(f"[Chain] edges 저장 오류: {e_err}")

                # ── 체인 노드 → article_nouns (별도 Gemini 호출 없이 고품질 노드 확보) ──
                title = art["title"] or ""
                for node in all_nodes:
                    position = "title" if node in title else "body"
                    try:
                        before_n = conn.total_changes
                        conn.execute(
                            """INSERT OR IGNORE INTO article_nouns
                               (article_id, noun, position) VALUES (?, ?, ?)""",
                            (art["id"], node, position),
                        )
                        total_nouns += 1
                        if conn.total_changes > before_n:
                            new_nodes += 1
                    except Exception:
                        pass

            print(f"  [{i+1}/{len(articles)}] article_id={art['id']} "
                  f"→ 체인 {len(chains)}개 / 엣지 {len(edges)}개 / 노드 {len(all_nodes)}개")

        else:
            # 체인이 없어도 "처리됨" 마킹 (재처리 방지)
            with get_db() as conn:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO causal_chains
                           (article_id, category, chain_text, confidence)
                           VALUES (?, ?, ?, ?)""",
                        (art["id"], "__none__", "__none__", 0),
                    )
                except Exception:
                    pass
            print(f"  [{i+1}/{len(articles)}] article_id={art['id']} → 체인 없음")

        processed += 1

        # 무료 10 RPM 제한 준수 (마지막 기사 이후는 대기 불필요)
        if i < len(articles) - 1:
            time.sleep(rate_limit_sec)

    remaining = 0
    try:
        with get_db() as conn:
            remaining = conn.execute("""
                SELECT COUNT(*) as c FROM articles
                WHERE id NOT IN (
                    SELECT DISTINCT article_id FROM causal_chains
                    WHERE article_id IS NOT NULL
                )
                AND body IS NOT NULL AND length(body) > 100
            """).fetchone()["c"]
    except Exception:
        pass

    return {
        "processed":         processed,
        "chains_saved":      total_chains,
        "edges_saved":       total_edges,
        "new_edges":         new_edges,         # 그래프에 처음 추가된 연결
        "reinforced_edges":  reinforced_edges,  # 이미 있던 연결에 증거 누적
        "new_nodes":         new_nodes,         # 처음 등장한 개념 노드
        "nouns_saved":       total_nouns,
        "remaining":         remaining,
    }


# ══════════════════════════════════════════════════════════════
#  질의 함수
# ══════════════════════════════════════════════════════════════

def _build_edge_rows(days: int = None) -> list:
    """
    causal_edges에서 (cause, effect, category, strength, evidence) 로우를 가져옴.
    days가 지정되면 해당 기간 내 기사에서 추출된 엣지만 반환.
    days=None → 전체 (기존 동작 유지)
    """
    with get_db() as conn:
        if days:
            rows = conn.execute("""
                SELECT ce.cause, ce.effect, ce.category,
                       SUM(ce.strength)  AS total_strength,
                       COUNT(*)          AS evidence_count
                FROM causal_edges ce
                JOIN articles a ON ce.article_id = a.id
                WHERE ce.category != '__none__'
                  AND a.published_at >= date('now', '-' || ? || ' days')
                GROUP BY ce.cause, ce.effect
            """, (days,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT cause, effect, category,
                       SUM(strength)  AS total_strength,
                       COUNT(*)       AS evidence_count
                FROM causal_edges
                WHERE category != '__none__'
                GROUP BY cause, effect
            """).fetchall()
    return rows


def get_impact_tree(trigger_noun: str, depth: int = 4, days: int = None) -> Dict:
    """
    "전쟁이 일어나면 무엇이 영향받나?" 형태의 순방향 영향 트리.

    trigger_noun에서 시작해 downstream 효과를 depth 단계까지 추적.
    각 레벨에서 evidence(등장 기사 수)와 strength(강도 합계)로 정렬.

    days: 최근 N일 기사 기반으로만 조회 (None=전체)
    """
    rows = _build_edge_rows(days=days)

    # 방향 그래프 구성
    forward: Dict[str, List] = {}
    for r in rows:
        forward.setdefault(r["cause"], []).append({
            "effect":   r["effect"],
            "category": r["category"],
            "strength": r["total_strength"],
            "evidence": r["evidence_count"],
        })

    def traverse(noun: str, depth: int, visited: set) -> List:
        if depth == 0 or noun in visited:
            return []
        visited = visited | {noun}
        children = sorted(
            forward.get(noun, []),
            key=lambda x: (x["evidence"], x["strength"]),
            reverse=True,
        )[:8]  # 레벨당 최대 8개 (과도한 팬아웃 방지)

        result = []
        for child in children:
            result.append({
                "noun":     child["effect"],
                "category": child["category"],
                "strength": child["strength"],
                "evidence": child["evidence"],
                "children": traverse(child["effect"], depth - 1, visited),
            })
        return result

    return {
        "root":    trigger_noun,
        "depth":   depth,
        "days":    days,
        "impacts": traverse(trigger_noun, depth, set()),
    }


def get_causal_chain(noun: str, depth: int = 3, direction: str = "both",
                     days: int = None) -> Dict:
    """
    기존 API 호환 유지.
    direction: "forward" | "backward" | "both"
    days: 최근 N일 기사 기반으로만 조회 (None=전체)
    """
    rows = _build_edge_rows(days=days)

    forward:  Dict[str, List] = {}
    backward: Dict[str, List] = {}

    for r in rows:
        entry = {
            "noun":     r["effect"],
            "category": r["category"],
            "strength": r["total_strength"],
            "evidence": r["evidence_count"],
        }
        forward.setdefault(r["cause"], []).append(entry)

        back_entry = {
            "noun":     r["cause"],
            "category": r["category"],
            "strength": r["total_strength"],
            "evidence": r["evidence_count"],
        }
        backward.setdefault(r["effect"], []).append(back_entry)

    def _traverse(noun: str, graph: Dict, depth: int, visited: set) -> List:
        if depth == 0 or noun in visited:
            return []
        visited = visited | {noun}
        children = sorted(
            graph.get(noun, []),
            key=lambda x: x["strength"],
            reverse=True,
        )[:5]
        return [{
            "noun":     c["noun"],
            "category": c["category"],
            "strength": c["strength"],
            "evidence": c["evidence"],
            "children": _traverse(c["noun"], graph, depth - 1, visited),
        } for c in children]

    result: Dict = {"root": noun, "depth": depth}
    if direction in ("forward", "both"):
        result["effects_chain"] = _traverse(noun, forward, depth, set())
    if direction in ("backward", "both"):
        result["causes_chain"]  = _traverse(noun, backward, depth, set())
    return result


def get_causal_path(noun_a: str, noun_b: str, max_depth: int = 6) -> Dict:
    """두 노드 간 인과 경로 BFS 탐색."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT cause, effect, category, SUM(strength) AS s
            FROM causal_edges
            WHERE category != '__none__'
            GROUP BY cause, effect
        """).fetchall()

    graph: Dict[str, List] = {}
    for r in rows:
        graph.setdefault(r["cause"], []).append({
            "noun": r["effect"], "category": r["category"], "strength": r["s"]
        })

    queue   = deque([(noun_a, [noun_a], [])])
    visited = {noun_a}

    while queue:
        current, path, hops = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for child in graph.get(current, []):
            n = child["noun"]
            new_path = path + [n]
            new_hops = hops + [{
                "from":     current,
                "to":       n,
                "category": child["category"],
                "strength": child["strength"],
            }]
            if n == noun_b:
                return {"path": new_path, "hops": new_hops, "length": len(new_hops)}
            if n not in visited:
                visited.add(n)
                queue.append((n, new_path, new_hops))

    return {"error": f"'{noun_a}'→'{noun_b}' 인과 경로를 찾을 수 없습니다"}


def get_category_chains(category: str, limit: int = 20) -> List[Dict]:
    """특정 카테고리의 대표 체인 조회."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT chain_text, confidence, COUNT(*) AS evidence
            FROM causal_chains
            WHERE category = ?
              AND chain_text != '__none__'
            GROUP BY chain_text
            ORDER BY evidence DESC, confidence DESC
            LIMIT ?
        """, (category, limit)).fetchall()
    return [dict(r) for r in rows]


def get_all_categories() -> List[Dict]:
    """카테고리별 체인 수 통계."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT category,
                   COUNT(*)       AS chain_count,
                   AVG(confidence) AS avg_confidence
            FROM causal_chains
            WHERE chain_text != '__none__'
            GROUP BY category
            ORDER BY chain_count DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_top_chains(limit: int = 30, days: int = None) -> List[Dict]:
    """
    증거 기사 수 기준 상위 체인 반환.
    days: 최근 N일 기사 기반으로만 집계 (None=전체)
    """
    with get_db() as conn:
        if days:
            rows = conn.execute("""
                SELECT cc.chain_text, cc.category, cc.confidence,
                       COUNT(*) AS evidence
                FROM causal_chains cc
                JOIN articles a ON cc.article_id = a.id
                WHERE cc.chain_text != '__none__'
                  AND a.published_at >= date('now', '-' || ? || ' days')
                GROUP BY cc.chain_text
                ORDER BY evidence DESC, confidence DESC
                LIMIT ?
            """, (days, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT chain_text, category, confidence, COUNT(*) AS evidence
                FROM causal_chains
                WHERE chain_text != '__none__'
                GROUP BY chain_text
                ORDER BY evidence DESC, confidence DESC
                LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_fresh_chains(days: int = 30, limit: int = 50) -> List[Dict]:
    """
    최근 N일 안에 처음 등장한 체인만 반환 (신규 인과 패턴 감지).

    "이번 주 뉴스에서 새로 나타난 인과관계"를 보여줌.
    기존에 이미 있던 체인은 제외 — 진짜 새로운 신호만.
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT cc.chain_text, cc.category, cc.confidence,
                   MIN(a.published_at) AS first_seen,
                   COUNT(*)            AS evidence
            FROM causal_chains cc
            JOIN articles a ON cc.article_id = a.id
            WHERE cc.chain_text != '__none__'
            GROUP BY cc.chain_text
            HAVING MIN(a.published_at) >= date('now', '-' || ? || ' days')
            ORDER BY first_seen DESC
            LIMIT ?
        """, (days, limit)).fetchall()
    return [dict(r) for r in rows]


def get_fading_chains(active_days: int = 30, fade_days: int = 90,
                      limit: int = 30) -> List[Dict]:
    """
    과거엔 자주 등장했지만 최근엔 사라진 체인 (식어가는 이슈).

    active_days 내에 등장 없고, fade_days 이내에는 등장했던 체인을 반환.
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT cc.chain_text, cc.category,
                   MAX(a.published_at) AS last_seen,
                   COUNT(*)            AS total_evidence
            FROM causal_chains cc
            JOIN articles a ON cc.article_id = a.id
            WHERE cc.chain_text != '__none__'
            GROUP BY cc.chain_text
            HAVING MAX(a.published_at) <  date('now', '-' || ? || ' days')
               AND MAX(a.published_at) >= date('now', '-' || ? || ' days')
            ORDER BY total_evidence DESC
            LIMIT ?
        """, (active_days, fade_days, limit)).fetchall()
    return [dict(r) for r in rows]


def get_multi_trigger_impacts(trigger_nouns: List[str], depth: int = 4,
                               days: int = None) -> Dict:
    """
    멀티 트리거 시뮬레이션: 여러 사건이 동시에 발생할 때 수렴하는 결과 탐색.

    예시: trigger_nouns=["미국 관세 인상", "중국 보복 조치", "유가 급등"]
    → 세 출발점에서 BFS를 각각 돌려, 공통으로 도달하는 노드를 수렴점으로 반환.

    days: 최근 N일 기사 기반으로만 조회 (None=전체)
    """
    rows = _build_edge_rows(days=days)

    # 방향 그래프 구성
    forward: Dict[str, List] = {}
    for r in rows:
        forward.setdefault(r["cause"], []).append({
            "effect":   r["effect"],
            "category": r["category"],
            "strength": r["total_strength"],
            "evidence": r["evidence_count"],
        })

    def bfs_reachable(start: str, max_depth: int) -> Dict[str, int]:
        """start에서 도달 가능한 노드와 최단 홉 수 반환."""
        dist   = {start: 0}
        queue  = deque([start])
        while queue:
            node = queue.popleft()
            if dist[node] >= max_depth:
                continue
            for child in forward.get(node, []):
                n = child["effect"]
                if n not in dist:
                    dist[n] = dist[node] + 1
                    queue.append(n)
        return dist

    if not trigger_nouns:
        return {"error": "trigger_nouns가 비어 있습니다"}

    # 각 트리거에서 도달 가능한 노드 집합 계산
    reachable_sets = [set(bfs_reachable(t, depth).keys()) for t in trigger_nouns]

    # 공통 도달 노드 (모든 트리거에서 도달 가능한 것)
    common = reachable_sets[0]
    for s in reachable_sets[1:]:
        common = common & s
    common -= set(trigger_nouns)  # 트리거 자체는 제외

    # 공통 노드를 증거 수 기준으로 정렬
    node_evidence: Dict[str, int] = {}
    for r in rows:
        node_evidence[r["effect"]] = node_evidence.get(r["effect"], 0) + r["evidence_count"]

    convergence = sorted(
        [{"noun": n, "evidence": node_evidence.get(n, 0)} for n in common],
        key=lambda x: x["evidence"],
        reverse=True,
    )[:20]

    # 트리거별 단독 도달 노드 (교집합에 없는 것 — 각 트리거의 고유 영향)
    unique_impacts = {}
    for i, t in enumerate(trigger_nouns):
        others = set()
        for j, s in enumerate(reachable_sets):
            if j != i:
                others |= s
        unique_impacts[t] = sorted(
            [{"noun": n, "evidence": node_evidence.get(n, 0)}
             for n in reachable_sets[i] - common - set(trigger_nouns)
             if n not in others],
            key=lambda x: x["evidence"],
            reverse=True,
        )[:10]

    return {
        "triggers":          trigger_nouns,
        "depth":             depth,
        "days":              days,
        "convergence":       convergence,       # 모든 트리거가 공통으로 향하는 결과
        "unique_impacts":    unique_impacts,    # 각 트리거만의 고유 영향
        "total_common":      len(convergence),
    }


# ══════════════════════════════════════════════════════════════
#  직접 실행
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  학습 데이터 관리
# ══════════════════════════════════════════════════════════════

def get_training_data_stats() -> Dict:
    """현재까지 저장된 학습 데이터 통계."""
    if not os.path.exists(TRAINING_JSONL):
        return {"count": 0, "file": TRAINING_JSONL, "size_kb": 0}

    count = 0
    try:
        with open(TRAINING_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        pass

    size_kb = round(os.path.getsize(TRAINING_JSONL) / 1024, 1)
    return {
        "count":   count,
        "file":    TRAINING_JSONL,
        "size_kb": size_kb,
    }


def export_training_data_alpaca(output_path: str = None) -> str:
    """
    ChatML JSONL → Alpaca 포맷 JSONL 변환 내보내기.
    Alpaca 포맷: {"instruction": "...", "input": "...", "output": "..."}
    일부 파인튜닝 프레임워크(Stanford Alpaca, LLaMA-Factory 등)에서 사용.
    """
    if output_path is None:
        output_path = os.path.join(TRAINING_DATA_DIR, "causal_chains_alpaca.jsonl")

    if not os.path.exists(TRAINING_JSONL):
        return "학습 데이터 없음"

    count = 0
    with open(TRAINING_JSONL, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                example  = json.loads(line)
                messages = example["messages"]
                system   = next(m["content"] for m in messages if m["role"] == "system")
                user     = next(m["content"] for m in messages if m["role"] == "user")
                asst     = next(m["content"] for m in messages if m["role"] == "assistant")

                alpaca = {
                    "instruction": system,
                    "input":       user,
                    "output":      asst,
                }
                fout.write(json.dumps(alpaca, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                continue

    return output_path


if __name__ == "__main__":
    import json as _json
    from database import init_db
    init_db()

    result = process_articles_for_chains(batch_size=5)
    print("체인 추출:", result)

    print("\n카테고리 통계:")
    for cat in get_all_categories():
        print(f"  {cat['category']}: {cat['chain_count']}개 체인")

    print("\n영향 트리 테스트 (중동 분쟁):")
    tree = get_impact_tree("중동 분쟁", depth=3)
    print(_json.dumps(tree, ensure_ascii=False, indent=2))
