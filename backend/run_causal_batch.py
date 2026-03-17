#!/usr/bin/env python3
"""
macro-graph 인과 체인 — Gemini Batch API 실행기

일반 API 대비 50% 할인, 터미널 닫아도 서버에서 계속 실행됨.
잡 ID만 저장해 두면 다음 날 다시 열어서 결과 수신 가능.

메뉴:
  [1] JSONL 생성   — 미처리 기사 → batch_input_YYYYMMDD_HHMMSS.jsonl
  [2] 배치 잡 제출  — JSONL 파일 업로드 후 잡 생성
  [3] 잡 상태 목록  — 추적 중인 잡들 현재 상태 출력
  [4] 결과 수신·저장 — 완료 잡 선택 → 다운로드 → DB 저장
  [q] 종료
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# ── 경로 설정 ──────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_FILE    = os.path.join(_BACKEND_DIR, "batch_jobs.json")   # 잡 추적 파일
INPUTS_DIR   = os.path.join(_BACKEND_DIR, "batch_inputs")      # JSONL 저장 디렉터리
os.makedirs(INPUTS_DIR, exist_ok=True)

# ── 환경변수 ───────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ══════════════════════════════════════════════════════════════
#  잡 추적 파일 (batch_jobs.json)
# ══════════════════════════════════════════════════════════════

def _load_jobs() -> Dict:
    """추적 중인 잡 목록 로드."""
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_jobs(jobs: Dict):
    """잡 목록 저장."""
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
#  프롬프트 (causal_extractor.py 의 CHAIN_PROMPT_V2 그대로 사용)
# ══════════════════════════════════════════════════════════════

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

VALID_CATEGORIES = {
    "실물경제", "지정학", "에너지시장", "공급망",
    "금융시장", "통화패권", "기술산업", "안보", "식량", "보건",
}


# ══════════════════════════════════════════════════════════════
#  [1] JSONL 생성
# ══════════════════════════════════════════════════════════════

def menu_create_jsonl():
    """미처리 기사를 Gemini Batch API 용 JSONL 파일로 변환."""
    from database import init_db, get_db
    init_db()

    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM articles").fetchone()["c"]
        rows = conn.execute("""
            SELECT a.id, a.title, a.body
            FROM articles a
            WHERE a.id NOT IN (
                SELECT DISTINCT article_id FROM causal_chains
                WHERE article_id IS NOT NULL
            )
            AND a.body IS NOT NULL AND length(a.body) > 100
            ORDER BY a.published_at DESC
        """).fetchall()

    remaining = len(rows)
    print(f"\n  전체 기사: {total:,}개  /  미처리: {remaining:,}개")

    if remaining == 0:
        print("  [완료] 처리할 기사가 없습니다.\n")
        return

    print(f"  몇 개의 기사를 JSONL에 포함할까요?")
    print(f"  (최대 {remaining:,}개, 전체 포함하려면 그냥 Enter)")
    raw = input("  > ").strip()

    if raw == "":
        limit = remaining
    elif raw.isdigit() and int(raw) > 0:
        limit = min(int(raw), remaining)
    else:
        print("  잘못된 입력입니다.")
        return

    articles = rows[:limit]

    # JSONL 작성
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(INPUTS_DIR, f"batch_input_{ts}.jsonl")

    article_ids = []
    with open(out_path, "w", encoding="utf-8") as f:
        for art in articles:
            text   = f"{art['title']}\n\n{(art['body'] or '')[:4000]}"
            prompt = CHAIN_PROMPT_V2.format(text=text)
            request_obj = {
                "contents": [
                    {"parts": [{"text": prompt}], "role": "user"}
                ]
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            article_ids.append(art["id"])

    # 매핑 파일 (JSONL과 같은 이름, .meta.json)
    meta_path = out_path.replace(".jsonl", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "created_at":  datetime.now().isoformat(),
            "article_ids": article_ids,
            "count":       len(article_ids),
        }, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n  ✓ JSONL 생성 완료")
    print(f"    파일:     {out_path}")
    print(f"    기사 수:  {len(article_ids):,}개")
    print(f"    파일 크기: {size_kb:.1f} KB")
    print(f"    메타:     {meta_path}")
    print(f"\n  → 다음: 메뉴 [2]에서 이 파일을 Gemini에 제출하세요.\n")


# ══════════════════════════════════════════════════════════════
#  [2] 배치 잡 제출
# ══════════════════════════════════════════════════════════════

def menu_submit_job():
    """JSONL 파일 업로드 후 Gemini Batch 잡 생성."""
    if not GEMINI_API_KEY:
        print("  [오류] GEMINI_API_KEY 환경변수가 없습니다.\n")
        return

    # 저장된 JSONL 파일 목록
    jsonl_files = sorted([
        f for f in os.listdir(INPUTS_DIR) if f.endswith(".jsonl")
    ], reverse=True)  # 최신 순

    if not jsonl_files:
        print("  [안내] batch_inputs/ 폴더에 JSONL 파일이 없습니다.")
        print("         먼저 메뉴 [1]로 JSONL을 생성하세요.\n")
        return

    print(f"\n  사용 가능한 JSONL 파일 ({len(jsonl_files)}개):")
    for i, fname in enumerate(jsonl_files):
        fpath    = os.path.join(INPUTS_DIR, fname)
        meta_path = fpath.replace(".jsonl", ".meta.json")
        count_str = ""
        if os.path.exists(meta_path):
            try:
                meta      = json.load(open(meta_path, encoding="utf-8"))
                count_str = f" ({meta['count']:,}개 기사)"
            except Exception:
                pass
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  [{i+1}] {fname}{count_str}  {size_kb:.1f} KB")

    raw = input("\n  제출할 파일 번호 선택 > ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(jsonl_files)):
        print("  잘못된 번호입니다.\n")
        return

    chosen_file = jsonl_files[int(raw) - 1]
    chosen_path = os.path.join(INPUTS_DIR, chosen_file)
    meta_path   = chosen_path.replace(".jsonl", ".meta.json")

    # 메타 파일 로드
    if not os.path.exists(meta_path):
        print(f"  [오류] 메타 파일이 없습니다: {meta_path}")
        print("         이 JSONL 파일에 대응하는 .meta.json 이 필요합니다.\n")
        return

    meta        = json.load(open(meta_path, encoding="utf-8"))
    article_ids = meta["article_ids"]
    count       = meta["count"]

    print(f"\n  선택: {chosen_file}  ({count:,}개 기사)")
    confirm = input("  제출하시겠습니까? [y/N] > ").strip().lower()
    if confirm != "y":
        print("  취소됐습니다.\n")
        return

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=GEMINI_API_KEY)

    # 1) 파일 업로드
    print(f"\n  [1/2] JSONL 파일 업로드 중...")
    try:
        uploaded = client.files.upload(file=chosen_path)
        print(f"    업로드 완료: {uploaded.name}")
    except Exception as e:
        print(f"  [오류] 파일 업로드 실패: {e}\n")
        return

    # 2) 배치 잡 생성
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"macro-causal-{ts}"

    print(f"  [2/2] 배치 잡 생성 중...")
    try:
        job = client.batches.create(
            model="gemini-2.5-flash",
            src=uploaded.name,
            config=genai_types.CreateBatchJobConfig(
                display_name=display_name,
            ),
        )
        print(f"    잡 생성 완료!")
        print(f"    job.name: {job.name}")
        print(f"    상태:     {job.state}")
    except Exception as e:
        print(f"  [오류] 배치 잡 생성 실패: {e}\n")
        return

    # 3) 잡 목록에 저장
    jobs = _load_jobs()
    jobs[job.name] = {
        "display_name":   display_name,
        "created_at":     datetime.now().isoformat(),
        "input_file":     chosen_file,
        "article_ids":    article_ids,
        "count":          count,
        "status":         str(job.state),
        "uploaded_file":  uploaded.name,
    }
    _save_jobs(jobs)

    print(f"\n  ✓ 완료! 잡 ID를 batch_jobs.json 에 저장했습니다.")
    print(f"  터미널을 닫아도 Gemini 서버에서 계속 실행됩니다.")
    print(f"  나중에 메뉴 [3] 또는 [4]로 상태 확인 및 결과 수신 가능합니다.\n")


# ══════════════════════════════════════════════════════════════
#  [3] 잡 상태 목록
# ══════════════════════════════════════════════════════════════

def menu_list_jobs():
    """추적 중인 잡들의 현재 상태를 Gemini에 쿼리해서 출력."""
    jobs = _load_jobs()

    if not jobs:
        print("\n  추적 중인 잡이 없습니다. 메뉴 [2]로 잡을 제출하세요.\n")
        return

    if not GEMINI_API_KEY:
        print("  [오류] GEMINI_API_KEY 환경변수가 없습니다.\n")
        return

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    print(f"\n  추적 중인 잡 ({len(jobs)}개):\n")
    print(f"  {'#':<3}  {'생성일시':<20}  {'기사수':>6}  {'상태':<25}  잡 ID")
    print("  " + "-" * 90)

    updated = False
    for idx, (job_name, info) in enumerate(jobs.items(), 1):
        created_str = info.get("created_at", "")[:16].replace("T", " ")
        count       = info.get("count", "?")

        # 실시간 상태 조회
        try:
            job      = client.batches.get(name=job_name)
            status   = str(job.state)
            # 상태 캐시 갱신
            if info.get("status") != status:
                jobs[job_name]["status"] = status
                updated = True
        except Exception as e:
            status = f"조회 오류: {e}"

        # 상태별 이모지
        status_icon = {
            "JOB_STATE_SUCCEEDED": "✓",
            "JOB_STATE_FAILED":    "✗",
            "JOB_STATE_CANCELLED": "⊘",
            "JOB_STATE_EXPIRED":   "⌛",
            "JOB_STATE_RUNNING":   "↻",
            "JOB_STATE_PENDING":   "…",
        }.get(status, "?")

        short_name = job_name.split("/")[-1] if "/" in job_name else job_name
        print(f"  [{idx:<2}]  {created_str:<20}  {count:>6}건  "
              f"{status_icon} {status:<22}  {short_name}")

    if updated:
        _save_jobs(jobs)

    print()
    print("  ※ JOB_STATE_SUCCEEDED → 메뉴 [4]로 결과를 받아 DB에 저장할 수 있습니다.")
    print()


# ══════════════════════════════════════════════════════════════
#  [4] 결과 수신 및 DB 저장
# ══════════════════════════════════════════════════════════════

def _parse_response_text(raw_text: str) -> list:
    """Gemini 응답 텍스트 → 유효한 체인 리스트로 파싱 (causal_extractor 로직 재사용)."""
    raw = raw_text.strip()

    # 마크다운 코드블록 제거 (방어 처리)
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
        raw_cat     = str(c.get("category", "기타")).strip()
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
            "extractor_version": "v2-batch",
        })

    return valid


def _save_chains_to_db(chains: list, article_id: int) -> Dict:
    """체인·엣지·노드를 DB에 저장. causal_extractor 의 저장 로직과 동일."""
    from database import get_db

    for c in chains:
        c["article_id"] = article_id

    # chains_to_edges 인라인 구현 (causal_extractor 의 chains_to_edges 와 동일)
    edges = []
    for c in chains:
        nodes      = c["chain"]
        chain_text = " → ".join(nodes)
        ev         = c.get("extractor_version", "v2-batch")
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

    new_chains     = 0
    new_edges      = 0
    reinforced     = 0
    article_nouns  = 0

    with get_db() as conn:
        # 체인 원본 저장
        for c in chains:
            chain_text = " → ".join(c["chain"])
            ev = c.get("extractor_version", "v2-batch")
            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO causal_chains
                   (article_id, category, chain_text, confidence, extractor_version)
                   VALUES (?, ?, ?, ?, ?)""",
                (c["article_id"], c["category"], chain_text, c["confidence"], ev),
            )
            if conn.total_changes > before:
                new_chains += 1

        # 엣지 저장
        for e in edges:
            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO causal_edges
                   (cause, effect, relation, strength,
                    article_id, category, chain_text, extractor_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (e["cause"], e["effect"], e["relation"], e["strength"],
                 e["article_id"], e["category"], e["chain_text"], e["extractor_version"]),
            )
            if conn.total_changes > before:
                new_edges += 1
            else:
                reinforced += 1

        # 체인 노드 → article_nouns 에도 저장 (graph_builder 에서 사용)
        all_nodes = set()
        for c in chains:
            for node in c["chain"]:
                all_nodes.add(node)

        for noun in all_nodes:
            conn.execute(
                """INSERT OR IGNORE INTO article_nouns
                   (article_id, noun, position)
                   VALUES (?, ?, 'body')""",
                (article_id, noun),
            )
            article_nouns += 1

    return {
        "new_chains":    new_chains,
        "new_edges":     new_edges,
        "reinforced":    reinforced,
        "article_nouns": article_nouns,
    }


def menu_receive_results():
    """완료된 잡을 선택해 결과 다운로드 후 DB에 저장."""
    jobs = _load_jobs()

    if not jobs:
        print("\n  추적 중인 잡이 없습니다. 메뉴 [2]로 잡을 제출하세요.\n")
        return

    if not GEMINI_API_KEY:
        print("  [오류] GEMINI_API_KEY 환경변수가 없습니다.\n")
        return

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 완료된 잡만 필터링 (실시간 조회)
    print("\n  완료된 잡을 조회 중...")
    succeeded_jobs = []
    for job_name, info in jobs.items():
        try:
            job    = client.batches.get(name=job_name)
            status = str(job.state)
            jobs[job_name]["status"] = status

            if status == "JOB_STATE_SUCCEEDED":
                succeeded_jobs.append((job_name, info, job))
        except Exception as e:
            print(f"  [경고] {job_name} 조회 실패: {e}")

    _save_jobs(jobs)

    if not succeeded_jobs:
        print("  현재 완료된(JOB_STATE_SUCCEEDED) 잡이 없습니다.")
        print("  메뉴 [3]으로 상태를 확인하세요.\n")
        return

    print(f"\n  완료된 잡 ({len(succeeded_jobs)}개):\n")
    for i, (job_name, info, _) in enumerate(succeeded_jobs, 1):
        created_str = info.get("created_at", "")[:16].replace("T", " ")
        count       = info.get("count", "?")
        short_name  = job_name.split("/")[-1] if "/" in job_name else job_name
        print(f"  [{i}] {created_str}  {count}건  {short_name}")

    raw = input("\n  수신할 잡 번호 선택 > ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(succeeded_jobs)):
        print("  잘못된 번호입니다.\n")
        return

    chosen_job_name, chosen_info, chosen_job = succeeded_jobs[int(raw) - 1]
    article_ids = chosen_info["article_ids"]
    count       = chosen_info["count"]

    print(f"\n  선택: {chosen_job_name.split('/')[-1]}  ({count}건)")

    # 결과 JSONL 다운로드
    print("  [1/3] 결과 파일 다운로드 중...")
    try:
        dest = chosen_job.dest
        # dest 객체에서 파일명 추출 (SDK 버전별 속성명 다름)
        result_file_name = (
            getattr(dest, "file_name", None)
            or getattr(dest, "fileName", None)
            or getattr(dest, "name", None)
        )
        if not result_file_name:
            print(f"  [오류] 결과 파일 이름을 찾을 수 없습니다. dest: {dest}\n")
            return

        print(f"    결과 파일: {result_file_name}")
        content_bytes = client.files.download(file=result_file_name)
        content_str   = content_bytes.decode("utf-8")
    except Exception as e:
        print(f"  [오류] 다운로드 실패: {e}\n")
        return

    # 결과 JSONL 저장 (나중에 재처리 가능하도록)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name  = chosen_job_name.split("/")[-1] if "/" in chosen_job_name else chosen_job_name
    result_path = os.path.join(INPUTS_DIR, f"batch_result_{short_name[:30]}_{ts}.jsonl")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(content_str)
    print(f"    저장: {result_path}")

    # 파싱 + DB 저장
    print("  [2/3] 응답 파싱 및 DB 저장 중...")
    from database import init_db
    init_db()

    lines = [l for l in content_str.splitlines() if l.strip()]
    total_lines    = len(lines)
    ok             = 0
    parse_fail     = 0
    empty_response = 0
    total_chains   = 0
    total_new_edges= 0
    total_reinforced = 0

    for i, line in enumerate(lines):
        # article_id 매핑 (순서 기반)
        if i >= len(article_ids):
            print(f"  [경고] 결과 줄 수({total_lines})가 기사 ID 수({len(article_ids)})를 초과합니다. 줄 {i} 건너뜀.")
            break

        article_id = article_ids[i]

        try:
            resp_obj = json.loads(line)
        except json.JSONDecodeError:
            parse_fail += 1
            continue

        # 응답 텍스트 추출
        # Gemini batch 출력 구조: {"response": {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}}
        try:
            status_field = resp_obj.get("status", "")
            if status_field and status_field != "OK":
                empty_response += 1
                continue

            candidates = (
                resp_obj.get("response", {})
                        .get("candidates", [])
            )
            if not candidates:
                empty_response += 1
                continue

            text = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            parse_fail += 1
            continue

        # 체인 파싱
        try:
            chains = _parse_response_text(text)
        except Exception:
            parse_fail += 1
            continue

        if not chains:
            empty_response += 1
            continue

        # DB 저장
        result = _save_chains_to_db(chains, article_id)
        ok             += 1
        total_chains   += result["new_chains"]
        total_new_edges+= result["new_edges"]
        total_reinforced += result["reinforced"]

        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{total_lines} 처리 중 (저장: {ok}건, 실패: {parse_fail}건)")

    # 잡 상태 업데이트 (received 마킹)
    jobs[chosen_job_name]["received_at"] = datetime.now().isoformat()
    jobs[chosen_job_name]["result_file"] = result_path
    _save_jobs(jobs)

    print(f"\n  [3/3] 완료!")
    print(f"  ┌ 응답 총 줄 수     : {total_lines}")
    print(f"  ├ 저장 성공         : {ok}건")
    print(f"  ├ 빈 응답 / 건너뜀  : {empty_response}건")
    print(f"  └ 파싱 실패         : {parse_fail}건")
    print(f"  ┌ 신규 체인         : {total_chains}개")
    print(f"  ├ 신규 엣지         : {total_new_edges}개")
    print(f"  └ 기존 강화         : {total_reinforced}개")
    print(f"\n  → 이제 API 서버의 /api/pipeline/build-graph 를 실행하면")
    print(f"    인과 그래프에 새 데이터가 반영됩니다.\n")


# ══════════════════════════════════════════════════════════════
#  메인 메뉴
# ══════════════════════════════════════════════════════════════

def print_menu():
    jobs   = _load_jobs()
    counts = {"SUCCEEDED": 0, "RUNNING": 0, "total": len(jobs)}
    for info in jobs.values():
        s = info.get("status", "")
        if "SUCCEEDED" in s: counts["SUCCEEDED"] += 1
        if "RUNNING"   in s: counts["RUNNING"]   += 1

    print("\n" + "=" * 55)
    print("  macro-graph  Gemini Batch API 인과 체인 추출기")
    print("=" * 55)
    print(f"  추적 중인 잡: {counts['total']}개 "
          f"(완료: {counts['SUCCEEDED']}  실행중: {counts['RUNNING']})")
    print()
    print("  [1] JSONL 생성     — 미처리 기사 → 배치 입력 파일")
    print("  [2] 배치 잡 제출   — JSONL → Gemini 서버로 전송")
    print("  [3] 잡 상태 목록   — 현재 진행 상황 확인")
    print("  [4] 결과 수신·저장 — 완료된 잡 → 다운로드 → DB 저장")
    print("  [q] 종료")
    print("=" * 55)


def main():
    if not GEMINI_API_KEY:
        print("[경고] GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("       .env 파일 또는 환경변수를 확인하세요.")

    while True:
        print_menu()
        choice = input("  선택 > ").strip().lower()

        if choice == "1":
            menu_create_jsonl()
        elif choice == "2":
            menu_submit_job()
        elif choice == "3":
            menu_list_jobs()
        elif choice == "4":
            menu_receive_results()
        elif choice in ("q", "quit", "exit", "종료"):
            print("\n  종료합니다.\n")
            break
        else:
            print("  [1] [2] [3] [4] 또는 [q] 중에서 선택하세요.\n")


if __name__ == "__main__":
    main()
