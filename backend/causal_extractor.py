"""
Gemini 기반 인과 체인 추출기 (v2 — 체인 중심 설계).

핵심 변경:
  - 기존: (cause, effect) 쌍 단위 추출 → 최대 5개 제한, 맥락 손실
  - 신규: 카테고리별 전체 체인 추출 → 제한 없음, 체인 원본 보존

목표: "전쟁이 일어나면 무엇이 영향받나?" 형태의 질의에 직접 대응하는
      방향 있는 인과 지식 그래프 축적.

DB 구조:
  causal_chains: 체인 전체 원본 저장 (category, chain_text, confidence)
  causal_edges:  체인을 (cause → effect) 쌍으로 분해해 그래프 탐색에 활용
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

# ── 학습 데이터 저장 경로 ────────────────────────────────────────
_BACKEND_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR  = os.path.join(_BACKEND_DIR, "training_data")
TRAINING_JSONL     = os.path.join(TRAINING_DATA_DIR, "causal_chains.jsonl")   # ChatML 포맷
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)


def _save_training_example(article_text: str, raw_output: str, system_prompt: str) -> None:
    """
    Gemini 입출력 쌍을 ChatML 포맷 JSONL로 저장.

    포맷 (Llama 3 / Mistral / Qwen 등 범용):
    {
      "messages": [
        {"role": "system",    "content": "<시스템 프롬프트>"},
        {"role": "user",      "content": "<기사 텍스트>"},
        {"role": "assistant", "content": "<JSON 응답>"}
      ]
    }

    raw_output: Gemini가 실제로 반환한 텍스트 (마크다운 포함 원본)
                → 로컬 모델이 동일한 출력 형식을 학습
    """
    example = {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": article_text},
            {"role": "assistant", "content": raw_output},
        ]
    }
    try:
        with open(TRAINING_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [학습 데이터 저장 오류] {e}")

# ── 프롬프트 ────────────────────────────────────────────────────
CHAIN_PROMPT = """다음 뉴스 기사를 읽고, 인과 체인을 최대한 많이 추출해주세요.

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


# ══════════════════════════════════════════════════════════════
#  체인 추출
# ══════════════════════════════════════════════════════════════

def extract_chains_from_article(article_id: int, title: str, body: str,
                                max_retries: int = 4) -> List[Dict]:
    """
    기사 한 건에서 Gemini로 인과 체인 추출.
    Rate limit(429) 발생 시 지수 백오프로 최대 max_retries회 재시도.

    반환:
        [{"category": "실물경제", "chain": ["A","B","C"], "confidence": 3,
          "article_id": 1}, ...]
    """
    if not GEMINI_API_KEY:
        return []

    text   = f"{title}\n\n{body[:4000]}"
    prompt = CHAIN_PROMPT.format(text=text)

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw = resp.text.strip()

            # 마크다운 코드블록 제거
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
                category    = str(c.get("category", "기타")).strip()[:20]
                confidence  = int(c.get("confidence", 1))

                if len(chain_nodes) < 3:
                    continue
                cleaned = [n.strip()[:15] for n in chain_nodes if len(n.strip()) >= 2]
                if len(cleaned) < 3:
                    continue
                if len(set(cleaned)) != len(cleaned):
                    continue

                valid.append({
                    "category":   category,
                    "chain":      cleaned,
                    "confidence": min(max(confidence, 1), 3),
                    "article_id": article_id,
                })

            # ── 성공한 경우에만 학습 데이터 저장 ──
            # 실패·빈 응답은 노이즈이므로 저장 안 함
            if valid:
                _save_training_example(
                    article_text=text,
                    raw_output=raw,
                    system_prompt=CHAIN_PROMPT.split("기사:")[0].strip(),
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
    체인 내 연속 노드 쌍이 모두 엣지가 되며, chain_text를 태그로 유지.
    """
    edges = []
    for c in chains:
        nodes      = c["chain"]
        chain_text = " → ".join(nodes)
        for i in range(len(nodes) - 1):
            edges.append({
                "cause":      nodes[i],
                "effect":     nodes[i + 1],
                "category":   c["category"],
                "relation":   f"{nodes[i]}→{nodes[i+1]}"[:20],
                "strength":   c["confidence"],
                "article_id": c["article_id"],
                "chain_text": chain_text,
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

    total_chains = 0
    total_edges  = 0
    total_nouns  = 0
    processed    = 0
    skipped      = 0

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
                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO causal_chains
                               (article_id, category, chain_text, confidence)
                               VALUES (?, ?, ?, ?)""",
                            (c["article_id"], c["category"],
                             chain_text, c["confidence"]),
                        )
                        total_chains += 1
                    except Exception as e:
                        print(f"[Chain] chains 저장 오류: {e}")

                # ── 엣지 저장 ──
                for e in edges:
                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO causal_edges
                               (cause, effect, relation, strength,
                                article_id, category, chain_text)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (e["cause"], e["effect"], e["relation"],
                             e["strength"], e["article_id"],
                             e["category"], e["chain_text"]),
                        )
                        total_edges += 1
                    except Exception as e_err:
                        print(f"[Chain] edges 저장 오류: {e_err}")

                # ── 체인 노드 → article_nouns (별도 Gemini 호출 없이 고품질 노드 확보) ──
                title = art["title"] or ""
                for node in all_nodes:
                    position = "title" if node in title else "body"
                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO article_nouns
                               (article_id, noun, position) VALUES (?, ?, ?)""",
                            (art["id"], node, position),
                        )
                        total_nouns += 1
                    except Exception:
                        pass

            print(f"  [{i+1}/{len(articles)}] article_id={art['id']} "
                  f"→ {len(chains)}개 체인 / {len(edges)}개 엣지 / {len(all_nodes)}개 노드")

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
        "processed":    processed,
        "chains_saved": total_chains,
        "edges_saved":  total_edges,
        "nouns_saved":  total_nouns,
        "remaining":    remaining,   # 아직 처리 안 된 기사 수
    }


# ══════════════════════════════════════════════════════════════
#  질의 함수
# ══════════════════════════════════════════════════════════════

def get_impact_tree(trigger_noun: str, depth: int = 4) -> Dict:
    """
    "전쟁이 일어나면 무엇이 영향받나?" 형태의 순방향 영향 트리.

    trigger_noun에서 시작해 downstream 효과를 depth 단계까지 추적.
    각 레벨에서 evidence(등장 기사 수)와 strength(강도 합계)로 정렬.
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT cause, effect, category,
                   SUM(strength)  AS total_strength,
                   COUNT(*)       AS evidence_count
            FROM causal_edges
            WHERE category != '__none__'
            GROUP BY cause, effect
        """).fetchall()

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
        "impacts": traverse(trigger_noun, depth, set()),
    }


def get_causal_chain(noun: str, depth: int = 3, direction: str = "both") -> Dict:
    """
    기존 API 호환 유지.
    direction: "forward" | "backward" | "both"
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT cause, effect, category,
                   SUM(strength) AS total_strength,
                   COUNT(*)      AS evidence_count
            FROM causal_edges
            WHERE category != '__none__'
            GROUP BY cause, effect
        """).fetchall()

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


def get_top_chains(limit: int = 30) -> List[Dict]:
    """증거 기사 수 기준 상위 체인 반환."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT chain_text, category, confidence, COUNT(*) AS evidence
            FROM causal_chains
            WHERE chain_text != '__none__'
            GROUP BY chain_text
            ORDER BY evidence DESC, confidence DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


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
