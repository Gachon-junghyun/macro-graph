"""
개념 정규화 레이어 (concept_normalizer.py)

핵심 설계 철학:
  추출 방식이 바뀌어도 기존 데이터를 버리지 않는다.

파이프라인:
  [추출] noun_extractor / causal_extractor
      ↓  (article_nouns 테이블 — raw 원본, 절대 수정 안 함)
  [정규화] concept_normalizer  ← 이 파일
      ↓  (concept_aliases 테이블 — raw_noun → canonical 매핑)
  [그래프] graph_builder
      ↓  (nodes / edges — canonical 기준으로 구축)

효과:
  "트럼프 관세" / "트럼프 관세 부과" / "미국 관세 정책"
  → 모두 canonical "트럼프 관세" 하나의 노드로 통합

호환성:
  - alias가 없는 raw_noun은 graph_builder에서 자기 자신을 canonical로 사용
  - self_map_all_unmapped() 호출 시 즉시 기존 데이터 1:1 매핑 (Gemini 불필요)
  - 이후 normalize_with_gemini()로 점진적 통합

DB 테이블:
  concepts        : canonical 개념 레지스트리 (id, canonical, category)
  concept_aliases : raw_noun → concept_id 매핑
"""

import json
import time
from typing import Dict, List, Optional, Set

from database import get_db
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Gemini 정규화 프롬프트 ─────────────────────────────────────────
NORMALIZE_PROMPT = """다음 경제·지정학 개념 목록에서 같은 의미이거나 거의 동일한 표현들을 묶어주세요.

개념 목록:
{nouns}

묶기 규칙:
- 같은 사건/정책의 다른 표기는 하나로 (예: "트럼프 관세" / "트럼프 관세 부과" / "미국 관세" → "트럼프 관세")
- 비슷해도 의미가 다른 개념은 분리 (예: "금리 인상" ≠ "금리 동결" ≠ "금리 인하")
- canonical: 가장 짧고 명확한 대표 표현 (2~12자), 목록 안에 있는 표현 우선
- 혼자인 개념도 반드시 포함 (aliases 빈 배열로)
- 목록에 있는 모든 개념이 반드시 결과에 포함되어야 함

JSON 배열로만 응답 (설명·마크다운 없이):
[
  {{"canonical": "대표표현", "aliases": ["원본1", "원본2"]}},
  {{"canonical": "혼자개념", "aliases": []}},
  ...
]"""


# ══════════════════════════════════════════════════════════════
#  내부 헬퍼
# ══════════════════════════════════════════════════════════════

def _upsert_canonical(conn, canonical: str, category: str = None) -> int:
    """concepts 테이블에 canonical을 INSERT OR IGNORE 후 id 반환."""
    conn.execute(
        "INSERT OR IGNORE INTO concepts (canonical, category) VALUES (?, ?)",
        (canonical, category),
    )
    row = conn.execute(
        "SELECT id FROM concepts WHERE canonical = ?", (canonical,)
    ).fetchone()
    return row["id"]


def _upsert_alias(conn, raw_noun: str, concept_id: int, method: str = "exact"):
    """concept_aliases 테이블에 raw_noun → concept_id 매핑 저장."""
    conn.execute(
        """INSERT OR IGNORE INTO concept_aliases (raw_noun, concept_id, method)
           VALUES (?, ?, ?)""",
        (raw_noun, concept_id, method),
    )


# ══════════════════════════════════════════════════════════════
#  즉시 호환성 확보 (Gemini 불필요)
# ══════════════════════════════════════════════════════════════

def self_map_all_unmapped(batch_size: int = 1000) -> Dict:
    """
    아직 concept_aliases에 없는 모든 raw_noun을 자기 자신으로 1:1 매핑.

    Gemini 없이 즉시 실행 가능 — 기존 article_nouns 데이터를 graph_builder가
    바로 활용할 수 있도록 호환성을 확보하는 첫 단계.
    이후 normalize_with_gemini()로 비슷한 개념들을 점진적으로 통합.
    """
    # 아직 매핑 안 된 raw_noun 수집
    with get_db() as conn:
        rows = conn.execute("""
            SELECT DISTINCT noun FROM article_nouns
            WHERE noun NOT GLOB '__*'
              AND noun NOT IN (SELECT raw_noun FROM concept_aliases)
        """).fetchall()

    unmapped = [r["noun"] for r in rows]
    if not unmapped:
        return {"mapped": 0, "message": "새로 매핑할 개념 없음"}

    saved = 0
    for i in range(0, len(unmapped), batch_size):
        batch = unmapped[i:i + batch_size]
        with get_db() as conn:
            for noun in batch:
                concept_id = _upsert_canonical(conn, noun)
                _upsert_alias(conn, noun, concept_id, method="self")
                saved += 1

    print(f"[self_map] {saved}개 개념 1:1 자기 매핑 완료")
    return {"mapped": saved}


# ══════════════════════════════════════════════════════════════
#  Gemini 기반 정규화 (유사 개념 통합)
# ══════════════════════════════════════════════════════════════

def _call_gemini_normalize(nouns: List[str]) -> List[Dict]:
    """
    Gemini에 개념 목록을 보내 유사 개념 그룹핑 결과 반환.
    반환: [{"canonical": "...", "aliases": ["...", ...]}, ...]
    """
    if not GEMINI_API_KEY or not nouns:
        return []

    noun_list_str = "\n".join(f"- {n}" for n in nouns)
    prompt = NORMALIZE_PROMPT.format(nouns=noun_list_str)

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    raw = resp.text.strip()

    # 마크다운 코드블록 제거
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```")[0].strip()

    result = json.loads(raw)
    if not isinstance(result, list):
        return []
    return result


def normalize_with_gemini(
    batch_size: int = 60,
    rate_limit_sec: float = 6.0,
    max_batches: int = 999,
) -> Dict:
    """
    아직 Gemini 정규화가 안 된 개념들을 배치로 묶어 통합.

    우선순위:
      1) self-mapped 개념(method='self') 중 비슷한 것들을 재그룹핑
      2) 아직 아무 매핑도 없는 신규 개념

    처리 흐름:
      raw_noun 배치 → Gemini → canonical 확정 → concept_aliases 업데이트
    """
    # self-mapped 또는 미매핑 개념 수집 (Gemini 정규화 대상)
    with get_db() as conn:
        rows = conn.execute("""
            SELECT DISTINCT noun FROM article_nouns
            WHERE noun NOT GLOB '__*'
              AND noun NOT IN (
                  SELECT raw_noun FROM concept_aliases WHERE method = 'gemini'
              )
            ORDER BY noun
        """).fetchall()

    candidates = [r["noun"] for r in rows]
    if not candidates:
        return {"normalized": 0, "message": "정규화할 개념 없음"}

    total_normalized = 0
    total_merged     = 0
    batches_done     = 0

    for i in range(0, len(candidates), batch_size):
        if batches_done >= max_batches:
            break

        batch = candidates[i:i + batch_size]
        batches_done += 1

        print(f"  [정규화 배치 {batches_done}] {len(batch)}개 개념 Gemini 처리 중...")

        try:
            groups = _call_gemini_normalize(batch)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                print(f"  [Rate Limit] 60초 대기 후 재시도...")
                time.sleep(60)
                try:
                    groups = _call_gemini_normalize(batch)
                except Exception as e2:
                    print(f"  [오류] 배치 건너뜀: {e2}")
                    continue
            else:
                print(f"  [오류] 배치 건너뜀: {e}")
                continue

        if not groups:
            continue

        # 결과 검증 및 DB 저장
        covered: Set[str] = set()
        with get_db() as conn:
            for group in groups:
                canonical = str(group.get("canonical", "")).strip()
                aliases   = group.get("aliases", [])

                if not canonical or len(canonical) < 2 or len(canonical) > 20:
                    continue

                concept_id = _upsert_canonical(conn, canonical)

                # canonical 자체도 alias로 등록
                _upsert_alias(conn, canonical, concept_id, method="gemini")
                covered.add(canonical)

                for alias in aliases:
                    alias = str(alias).strip()
                    if not alias or alias == canonical:
                        continue
                    # 기존 self-map을 gemini 매핑으로 업그레이드
                    conn.execute(
                        """INSERT OR REPLACE INTO concept_aliases
                           (raw_noun, concept_id, method)
                           VALUES (?, ?, 'gemini')""",
                        (alias, concept_id),
                    )
                    covered.add(alias)
                    total_merged += 1

            total_normalized += len(covered)

        # 배치에 있었지만 Gemini가 누락한 항목은 self-map으로 보장
        missed = set(batch) - covered
        if missed:
            with get_db() as conn:
                for noun in missed:
                    concept_id = _upsert_canonical(conn, noun)
                    _upsert_alias(conn, noun, concept_id, method="self")

        print(f"    → {len(covered)}개 처리, {total_merged}개 통합")

        if i + batch_size < len(candidates) and batches_done < max_batches:
            time.sleep(rate_limit_sec)

    return {
        "normalized": total_normalized,
        "merged":     total_merged,
        "batches":    batches_done,
    }


# ══════════════════════════════════════════════════════════════
#  조회 헬퍼 (graph_builder / api 에서 사용)
# ══════════════════════════════════════════════════════════════

def load_alias_map() -> Dict[str, str]:
    """
    concept_aliases 전체를 {raw_noun: canonical} dict로 반환.
    DB가 없거나 테이블이 없으면 빈 dict (기존 동작 유지).

    graph_builder.build_cooccurrence() 첫 줄에서 호출.
    """
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT ca.raw_noun, c.canonical
                FROM concept_aliases ca
                JOIN concepts c ON ca.concept_id = c.id
            """).fetchall()
        return {r["raw_noun"]: r["canonical"] for r in rows}
    except Exception:
        return {}


def get_canonical(raw_noun: str) -> str:
    """단일 raw_noun의 canonical 반환. 없으면 raw_noun 그대로."""
    try:
        with get_db() as conn:
            row = conn.execute("""
                SELECT c.canonical
                FROM concept_aliases ca
                JOIN concepts c ON ca.concept_id = c.id
                WHERE ca.raw_noun = ?
            """, (raw_noun,)).fetchone()
        return row["canonical"] if row else raw_noun
    except Exception:
        return raw_noun


def get_raw_nouns_for_canonical(canonical: str) -> List[str]:
    """
    특정 canonical로 매핑된 모든 raw_noun 목록 반환.
    api.py의 노드 상세 조회(관련 기사 검색)에서 사용.
    canonical 자체도 포함.
    """
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT ca.raw_noun
                FROM concept_aliases ca
                JOIN concepts c ON ca.concept_id = c.id
                WHERE c.canonical = ?
            """, (canonical,)).fetchall()
        result = [r["raw_noun"] for r in rows]
        if canonical not in result:
            result.append(canonical)
        return result
    except Exception:
        return [canonical]


# ══════════════════════════════════════════════════════════════
#  통계
# ══════════════════════════════════════════════════════════════

def get_stats() -> Dict:
    """정규화 현황 통계."""
    try:
        with get_db() as conn:
            total_raw      = conn.execute(
                "SELECT COUNT(DISTINCT noun) as c FROM article_nouns WHERE noun NOT GLOB '__*'"
            ).fetchone()["c"]
            total_concepts = conn.execute(
                "SELECT COUNT(*) as c FROM concepts"
            ).fetchone()["c"]
            total_aliases  = conn.execute(
                "SELECT COUNT(*) as c FROM concept_aliases"
            ).fetchone()["c"]
            gemini_mapped  = conn.execute(
                "SELECT COUNT(*) as c FROM concept_aliases WHERE method = 'gemini'"
            ).fetchone()["c"]
            self_mapped    = conn.execute(
                "SELECT COUNT(*) as c FROM concept_aliases WHERE method = 'self'"
            ).fetchone()["c"]
            unmapped       = conn.execute("""
                SELECT COUNT(DISTINCT noun) as c FROM article_nouns
                WHERE noun NOT GLOB '__*'
                  AND noun NOT IN (SELECT raw_noun FROM concept_aliases)
            """).fetchone()["c"]

        compression = round(total_aliases / total_concepts, 2) if total_concepts > 0 else 1.0
        return {
            "total_raw_nouns":  total_raw,
            "total_concepts":   total_concepts,
            "total_aliases":    total_aliases,
            "gemini_mapped":    gemini_mapped,
            "self_mapped":      self_mapped,
            "unmapped":         unmapped,
            "compression_ratio": compression,  # aliases/concepts — 높을수록 통합 효과 큼
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════
#  직접 실행
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from database import init_db
    init_db()

    print("=== 정규화 현황 ===")
    stats = get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n[1단계] self_map — 기존 데이터 즉시 호환성 확보...")
    result = self_map_all_unmapped()
    print(f"  → {result}")

    print("\n[2단계] Gemini 정규화 (소규모 테스트: max_batches=2)...")
    result = normalize_with_gemini(batch_size=40, max_batches=2)
    print(f"  → {result}")

    print("\n=== 정규화 후 통계 ===")
    stats = get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
