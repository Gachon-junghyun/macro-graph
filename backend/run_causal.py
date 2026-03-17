#!/usr/bin/env python3
"""
macro-graph 인과 체인 자동 추출 실행기

무료 Gemini AI Studio 제한:
  - 10 RPM  → 기사 간 7초 대기
  - 250 RPD → 하루 245회로 제한 (여유 5회)

Ctrl+C → 현재 배치 완료 후 안전 종료 (이미 처리된 기사는 모두 보존)
재실행 → 이어서 진행 (처리된 기사 자동 제외)
"""

import json
import os
import signal
import sys
import time
from datetime import date, datetime

# ── 설정 ──────────────────────────────────────────────────────────
DAILY_LIMIT    = 3000        # 250 RPD 중 5개 여유
BATCH_SIZE     = 10         # 한 번에 처리할 기사 수
RATE_LIMIT_SEC = 7.0        # 기사 간 대기 (10 RPM = 6초, 여유 1초)
COUNTER_FILE   = os.path.join(os.path.dirname(__file__), ".daily_counter.json")

# ── Ctrl+C 핸들러 ─────────────────────────────────────────────────
_stop_requested = False

def _handle_sigint(sig, frame):
    global _stop_requested
    print("\n\n[중단 요청] 현재 배치 완료 후 종료합니다... (재실행하면 이어서 진행)")
    _stop_requested = True

signal.signal(signal.SIGINT, _handle_sigint)


# ── 일별 카운터 ───────────────────────────────────────────────────

def _load_counter() -> dict:
    """오늘 날짜 기준 카운터 로드. 날짜 바뀌면 자동 리셋."""
    today = str(date.today())
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {"date": today, "count": 0}


def _save_counter(counter: dict):
    try:
        with open(COUNTER_FILE, "w") as f:
            json.dump(counter, f)
    except Exception as e:
        print(f"[카운터 저장 오류] {e}")


# ── DB 상태 조회 ──────────────────────────────────────────────────

def _get_remaining_articles() -> int:
    """아직 인과 체인 추출이 안 된 기사 수."""
    from database import get_db
    with get_db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as c FROM articles
            WHERE id NOT IN (
                SELECT DISTINCT article_id FROM causal_chains
                WHERE article_id IS NOT NULL
            )
            AND body IS NOT NULL AND length(body) > 100
        """).fetchone()
    return row["c"]


def _get_db_stats() -> dict:
    """현재 DB 누적 통계."""
    from database import get_db
    with get_db() as conn:
        articles   = conn.execute("SELECT COUNT(*) as c FROM articles").fetchone()["c"]
        chains     = conn.execute(
            "SELECT COUNT(*) as c FROM causal_chains WHERE chain_text != '__none__'"
        ).fetchone()["c"]
        edges      = conn.execute(
            "SELECT COUNT(*) as c FROM causal_edges"
        ).fetchone()["c"]
        nouns      = conn.execute(
            "SELECT COUNT(*) as c FROM article_nouns WHERE noun NOT GLOB '__*'"
        ).fetchone()["c"]
    return {"articles": articles, "chains": chains, "edges": edges, "nouns": nouns}


# ── 메인 ─────────────────────────────────────────────────────────

def main():
    from database import init_db
    from causal_extractor import process_articles_for_chains

    init_db()

    counter    = _load_counter()
    today_used = counter["count"]
    remaining  = _get_remaining_articles()
    stats      = _get_db_stats()

    print("=" * 58)
    print("  macro-graph 인과 체인 추출기")
    print("=" * 58)
    print(f"  [DB 현황]")
    print(f"    전체 기사        : {stats['articles']:,}개")
    print(f"    미처리 기사      : {remaining:,}개")
    print(f"    누적 체인        : {stats['chains']:,}개")
    print(f"    누적 엣지        : {stats['edges']:,}개")
    print(f"    누적 개념(노드)  : {stats['nouns']:,}개")
    print(f"  [오늘 사용량]")
    print(f"    사용 / 한도      : {today_used} / {DAILY_LIMIT}")
    print(f"    잔여 요청        : {DAILY_LIMIT - today_used}회")
    print(f"  [설정]")
    print(f"    배치 크기        : {BATCH_SIZE}개")
    print(f"    기사 간 대기     : {RATE_LIMIT_SEC}초")
    print(f"  Ctrl+C → 현재 배치 끝나고 안전 종료")
    print("=" * 58)

    # ── 한도 초과 / 처리 없음 사전 체크 ──
    if today_used >= DAILY_LIMIT:
        print(f"\n[종료] 오늘 일별 한도({DAILY_LIMIT}회) 도달.")
        print(f"       내일 자정 이후 다시 실행하면 자동으로 이어서 진행됩니다.")
        return

    if remaining == 0:
        print("\n[완료] 처리할 기사가 없습니다. 모두 처리됐어요!")
        return

    # ── 루프 ──────────────────────────────────────────────────────
    total_processed = 0
    total_chains    = 0
    total_edges     = 0
    total_nouns     = 0
    batch_num       = 0
    start_time      = time.time()

    while not _stop_requested:
        today_remaining = DAILY_LIMIT - today_used
        if today_remaining <= 0:
            print(f"\n[종료] 오늘 일별 한도({DAILY_LIMIT}회) 도달.")
            print(f"       내일 자정 이후 재실행하면 이어서 진행됩니다.")
            break

        this_batch = min(BATCH_SIZE, today_remaining)
        batch_num += 1
        elapsed    = int(time.time() - start_time)
        elapsed_str = f"{elapsed//60}분 {elapsed%60}초"

        print(f"\n[배치 {batch_num}] {this_batch}개 처리 시작  "
              f"(오늘 잔여: {today_remaining}회 | 경과: {elapsed_str})")

        result = process_articles_for_chains(
            batch_size=this_batch,
            rate_limit_sec=RATE_LIMIT_SEC,
        )

        processed_now = result.get("processed", 0)
        chains_now    = result.get("chains_saved", 0)
        edges_now     = result.get("edges_saved", 0)
        nouns_now     = result.get("nouns_saved", 0)
        remaining_now = result.get("remaining", 0)

        total_processed += processed_now
        total_chains    += chains_now
        total_edges     += edges_now
        total_nouns     += nouns_now
        today_used      += processed_now

        # 카운터 즉시 저장 (Ctrl+C 후에도 유지)
        counter["count"] = today_used
        _save_counter(counter)

        print(f"  → 처리: {processed_now}개 | 체인: {chains_now}개 | "
              f"엣지: {edges_now}개 | 개념: {nouns_now}개")
        print(f"  → 오늘 누적: {today_used} / {DAILY_LIMIT}  |  미처리 잔여: {remaining_now}개")

        # 처리할 기사 없음 → 종료
        if remaining_now == 0 or result.get("message") == "처리할 기사 없음":
            print("\n[완료] 모든 기사 처리 완료!")
            break

        if _stop_requested:
            print("here")
            break

        # 배치 간 짧은 추가 대기
        if not _stop_requested:
            time.sleep(2)

    # ── 최종 요약 ─────────────────────────────────────────────────
    elapsed_total = int(time.time() - start_time)
    final_stats   = _get_db_stats()

    print("\n" + "=" * 58)
    print("  이번 실행 결과")
    print("=" * 58)
    print(f"  처리한 기사  : {total_processed}개")
    print(f"  저장된 체인  : {total_chains}개")
    print(f"  저장된 엣지  : {total_edges}개")
    print(f"  저장된 개념  : {total_nouns}개")
    print(f"  소요 시간    : {elapsed_total//60}분 {elapsed_total%60}초")
    print(f"  오늘 총 사용 : {today_used} / {DAILY_LIMIT}")
    print(f"  [DB 누적 현황]")
    print(f"    체인 : {final_stats['chains']:,}개  |  엣지 : {final_stats['edges']:,}개  |  개념 : {final_stats['nouns']:,}개")
    print("=" * 58)
    print("\n  재실행하면 이어서 진행됩니다.")


if __name__ == "__main__":
    main()
