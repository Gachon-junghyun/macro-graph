#!/usr/bin/env python3
"""
v1 vs v2 인과 체인 추출 비교 테스트.

- 기존에 처리된 기사 5개를 선택해 v2 설정(thinking=0, 강화 프롬프트)으로 재추출
- v1 결과(DB에서 읽기)와 v2 결과(실시간 추출)를 나란히 출력
- 토큰 사용량도 함께 출력
- v2 결과는 results_v2/ 폴더에 JSON으로 저장 (DB에는 쓰지 않음 — 검토 후 결정)

실행:
  cd backend && python compare_v1_v2.py
"""

import os
import sys
import json
import time
import sqlite3

from dotenv import load_dotenv
load_dotenv()

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BACKEND_DIR)

RESULTS_V1_DIR = os.path.join(_BACKEND_DIR, "results_v1")
RESULTS_V2_DIR = os.path.join(_BACKEND_DIR, "results_v2")
os.makedirs(RESULTS_V2_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── v2 프롬프트 (causal_extractor.py와 동기화) ────────────────────
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
- 동일 기사에서 최소 3개, 최대 10개 체인 추출
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


def _extract_v2(article_id: int, title: str, body: str) -> dict:
    """v2 설정으로 추출 (thinking_budget=0, 강화 프롬프트). DB 저장 없이 결과만 반환."""
    from google import genai
    from google.genai import types as genai_types

    text   = f"{title}\n\n{body[:4000]}"
    prompt = CHAIN_PROMPT_V2.format(text=text)

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
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

        # 토큰 사용량 수집
        usage = getattr(resp, "usage_metadata", None)
        token_info = {}
        if usage:
            token_info = {
                "prompt_tokens":    getattr(usage, "prompt_token_count", None),
                "output_tokens":    getattr(usage, "candidates_token_count", None),
                "thinking_tokens":  getattr(usage, "thoughts_token_count", None),
                "total_tokens":     getattr(usage, "total_token_count", None),
            }

        # 마크다운 코드블록 방어 제거
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()

        chains_raw = json.loads(raw)
        valid = []
        for c in chains_raw:
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
                "category":          category,
                "chain":             cleaned,
                "confidence":        min(max(confidence, 1), 3),
                "article_id":        article_id,
                "extractor_version": "v2",
            })

        return {"chains": valid, "tokens": token_info, "raw": raw, "error": None}

    except Exception as e:
        return {"chains": [], "tokens": {}, "raw": "", "error": str(e)}


def _get_v1_chains(conn: sqlite3.Connection, article_id: int) -> list:
    """DB에서 v1 체인 읽기."""
    rows = conn.execute(
        """SELECT category, chain_text, confidence
           FROM causal_chains
           WHERE article_id = ? AND chain_text != '__none__'
             AND (extractor_version = 'v1' OR extractor_version IS NULL)
           ORDER BY confidence DESC""",
        (article_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def _print_comparison(article_id: int, title: str,
                       v1_chains: list, v2_result: dict) -> None:
    """v1 / v2 결과를 나란히 출력."""
    sep = "─" * 70
    v2_chains  = v2_result["chains"]
    tokens     = v2_result["tokens"]
    error      = v2_result["error"]

    print(f"\n{'═'*70}")
    print(f"  기사 ID : {article_id}")
    print(f"  제목    : {title[:60]}")
    print(f"{'═'*70}")

    # ── v1 ──
    print(f"\n  [v1 결과] — {len(v1_chains)}개 체인 (DB 저장본)")
    print(f"  {sep}")
    for i, c in enumerate(v1_chains, 1):
        print(f"  {i:2d}. [{c['category']}] {c['chain_text']}  (conf={c['confidence']})")
    if not v1_chains:
        print("  (없음)")

    # ── v2 ──
    print(f"\n  [v2 결과] — {len(v2_chains)}개 체인 (thinking=0, 강화 프롬프트)")
    print(f"  {sep}")
    if error:
        print(f"  [오류] {error}")
    else:
        for i, c in enumerate(v2_chains, 1):
            chain_str = " → ".join(c["chain"])
            print(f"  {i:2d}. [{c['category']}] {chain_str}  (conf={c['confidence']})")
        if not v2_chains:
            print("  (없음)")

    # ── 토큰 ──
    if tokens:
        print(f"\n  [토큰 사용량 — v2]")
        print(f"    입력(prompt)  : {tokens.get('prompt_tokens', '-')}")
        print(f"    출력(output)  : {tokens.get('output_tokens', '-')}")
        print(f"    사고(thinking): {tokens.get('thinking_tokens', '-')}")
        print(f"    합계(total)   : {tokens.get('total_tokens', '-')}")

    # ── 차이 요약 ──
    delta = len(v2_chains) - len(v1_chains)
    sign  = "+" if delta >= 0 else ""
    print(f"\n  [요약] v1={len(v1_chains)}개 → v2={len(v2_chains)}개  ({sign}{delta})")


def main():
    if not GEMINI_API_KEY:
        print("[오류] GEMINI_API_KEY 없음")
        sys.exit(1)

    db_path = os.path.join(_BACKEND_DIR, "macro_graph.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 체인이 있는 기사 중 최근 5개 선택
    sample_articles = conn.execute("""
        SELECT DISTINCT a.id, a.title, a.body
        FROM articles a
        JOIN causal_chains cc ON a.id = cc.article_id
        WHERE cc.chain_text != '__none__'
          AND a.body IS NOT NULL AND length(a.body) > 100
        ORDER BY a.id DESC
        LIMIT 5
    """).fetchall()

    if not sample_articles:
        print("[오류] 샘플 기사를 찾을 수 없음")
        conn.close()
        return

    print("=" * 70)
    print("  v1 vs v2 인과 체인 추출 비교 테스트")
    print(f"  샘플 {len(sample_articles)}개  |  모델: gemini-2.5-flash  |  thinking_budget=0")
    print("=" * 70)

    all_results = []

    for idx, art in enumerate(sample_articles):
        article_id = art["id"]
        title      = art["title"] or ""
        body       = art["body"] or ""

        print(f"\n[{idx+1}/{len(sample_articles)}] article_id={article_id} 처리 중...")

        v1_chains  = _get_v1_chains(conn, article_id)
        v2_result  = _extract_v2(article_id, title, body)

        _print_comparison(article_id, title, v1_chains, v2_result)

        all_results.append({
            "article_id": article_id,
            "title":      title[:80],
            "v1_chains":  v1_chains,
            "v2_chains":  v2_result["chains"],
            "v2_tokens":  v2_result["tokens"],
            "v2_error":   v2_result["error"],
        })

        # API rate limit 방어 (마지막 기사는 대기 불필요)
        if idx < len(sample_articles) - 1:
            time.sleep(7)

    conn.close()

    # ── 결과 저장 ──────────────────────────────────────────────
    out_path = os.path.join(RESULTS_V2_DIR, "compare_v1_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  비교 결과 저장: {out_path}")
    print("=" * 70)

    # ── 전체 요약 ─────────────────────────────────────────────
    total_v1 = sum(len(r["v1_chains"]) for r in all_results)
    total_v2 = sum(len(r["v2_chains"]) for r in all_results)
    errors   = sum(1 for r in all_results if r["v2_error"])

    print(f"\n  [전체 요약]")
    print(f"    v1 총 체인 : {total_v1}개")
    print(f"    v2 총 체인 : {total_v2}개")
    print(f"    오류       : {errors}건")
    avg_delta = (total_v2 - total_v1) / len(all_results) if all_results else 0
    print(f"    기사당 평균 체인 변화: {avg_delta:+.1f}")

    # 토큰 합계
    total_tokens = sum(
        r["v2_tokens"].get("total_tokens") or 0 for r in all_results
    )
    if total_tokens:
        print(f"    v2 총 토큰 : {total_tokens:,}")


if __name__ == "__main__":
    main()
