"""4단계: 가격 데이터 결합 — 자산 노드 + 수익률 계산."""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

from database import get_db

# ── 자산 노드 화이트리스트 ─────────────────────────────────────
ASSET_NODES: Dict[str, Dict] = {
    # yfinance tickers
    "금": {"ticker": "GC=F", "provider": "yfinance", "type": "commodity"},
    "유가": {"ticker": "CL=F", "provider": "yfinance", "type": "commodity"},
    "달러": {"ticker": "DX=F", "provider": "yfinance", "type": "currency"},
    "엔화": {"ticker": "JPY=X", "provider": "yfinance", "type": "currency"},
    "나스닥": {"ticker": "^IXIC", "provider": "yfinance", "type": "index"},
    "S&P": {"ticker": "^GSPC", "provider": "yfinance", "type": "index"},
    "다우": {"ticker": "^DJI", "provider": "yfinance", "type": "index"},
    "비트코인": {"ticker": "BTC-USD", "provider": "yfinance", "type": "crypto"},
    # pykrx tickers
    "삼성전자": {"ticker": "005930", "provider": "pykrx", "type": "stock_kr"},
    "SK하이닉스": {"ticker": "000660", "provider": "pykrx", "type": "stock_kr"},
    "LG에너지솔루션": {"ticker": "373220", "provider": "pykrx", "type": "stock_kr"},
    "한화에어로": {"ticker": "012450", "provider": "pykrx", "type": "stock_kr"},
    "LIG넥스원": {"ticker": "079550", "provider": "pykrx", "type": "stock_kr"},
    "대한항공": {"ticker": "003490", "provider": "pykrx", "type": "stock_kr"},
    "현대차": {"ticker": "005380", "provider": "pykrx", "type": "stock_kr"},
    "POSCO홀딩스": {"ticker": "005490", "provider": "pykrx", "type": "stock_kr"},
    "코스피": {"ticker": "KOSPI", "provider": "pykrx_index", "type": "index_kr"},
    "코스닥": {"ticker": "KOSDAQ", "provider": "pykrx_index", "type": "index_kr"},
}


def fetch_prices(days_back: int = 180) -> Dict:
    """자산별 가격 데이터 수집 후 DB 저장."""
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    to_date = datetime.now().strftime("%Y%m%d")

    saved = 0
    errors = []

    for asset_name, info in ASSET_NODES.items():
        ticker = info["ticker"]
        provider = info["provider"]

        try:
            prices = []

            if provider == "yfinance":
                import yfinance as yf
                data = yf.download(
                    ticker, start=from_date[:4]+"-"+from_date[4:6]+"-"+from_date[6:],
                    end=to_date[:4]+"-"+to_date[4:6]+"-"+to_date[6:],
                    progress=False,
                )
                if data is not None and len(data) > 0:
                    for idx, row in data.iterrows():
                        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
                        close_val = float(row["Close"].iloc[0]) if hasattr(row["Close"], 'iloc') else float(row["Close"])
                        prices.append((date_str, close_val))

            elif provider == "pykrx":
                from pykrx import stock
                df = stock.get_market_ohlcv(from_date, to_date, ticker)
                if df is not None and len(df) > 0:
                    for idx, row in df.iterrows():
                        date_str = idx.strftime("%Y-%m-%d")
                        prices.append((date_str, float(row["종가"])))

            elif provider == "pykrx_index":
                from pykrx import stock
                if ticker == "KOSPI":
                    df = stock.get_index_ohlcv(from_date, to_date, "1001")
                elif ticker == "KOSDAQ":
                    df = stock.get_index_ohlcv(from_date, to_date, "2001")
                if df is not None and len(df) > 0:
                    for idx, row in df.iterrows():
                        date_str = idx.strftime("%Y-%m-%d")
                        prices.append((date_str, float(row["종가"])))

            # DB 저장
            with get_db() as conn:
                for date_str, close_price in prices:
                    conn.execute(
                        """INSERT OR REPLACE INTO price_data
                           (ticker, asset_name, date, close_price)
                           VALUES (?, ?, ?, ?)""",
                        (ticker, asset_name, date_str, close_price),
                    )
                    saved += 1

            print(f"  {asset_name}: {len(prices)}개 가격 저장")

        except Exception as e:
            errors.append(f"{asset_name}: {e}")
            print(f"  {asset_name} 오류: {e}")

    return {"saved": saved, "errors": errors}


def detect_mention_spikes() -> List[Dict]:
    """언급 급증일 감지 (전일 대비 3배 이상)."""
    spikes = []

    with get_db() as conn:
        # 자산 노드별로 날짜별 언급 횟수 계산
        for asset_name in ASSET_NODES:
            rows = conn.execute("""
                SELECT a.published_at as date, COUNT(*) as cnt
                FROM article_nouns an
                JOIN articles a ON an.article_id = a.id
                WHERE an.noun = ?
                GROUP BY a.published_at
                ORDER BY a.published_at
            """, (asset_name,)).fetchall()

            if len(rows) < 3:
                continue

            for i in range(2, len(rows)):
                current_count = rows[i]["cnt"]
                prev_avg = sum(rows[j]["cnt"] for j in range(max(0, i-7), i)) / min(i, 7)

                if prev_avg > 0 and current_count >= prev_avg * 3:
                    spike_date = rows[i]["date"]
                    spikes.append({
                        "noun": asset_name,
                        "spike_date": spike_date,
                        "mention_count": current_count,
                        "prev_avg_count": round(prev_avg, 1),
                    })

    return spikes


def calculate_returns_for_spikes(spikes: List[Dict]) -> List[Dict]:
    """급증일 기준 수익률 계산."""
    enriched = []

    with get_db() as conn:
        for spike in spikes:
            asset_info = ASSET_NODES.get(spike["noun"])
            if not asset_info:
                continue

            ticker = asset_info["ticker"]
            spike_date = spike["spike_date"]

            # 급증일 기준 가격
            base_price_row = conn.execute(
                """SELECT close_price FROM price_data
                   WHERE ticker = ? AND date >= ? ORDER BY date LIMIT 1""",
                (ticker, spike_date),
            ).fetchone()

            if not base_price_row:
                continue

            base_price = base_price_row["close_price"]

            # +1일, +7일, +30일 가격
            returns = {}
            for label, days in [("1d", 1), ("7d", 7), ("30d", 30)]:
                target_date = (
                    datetime.strptime(spike_date[:10], "%Y-%m-%d") + timedelta(days=days)
                ).strftime("%Y-%m-%d")

                future_row = conn.execute(
                    """SELECT close_price FROM price_data
                       WHERE ticker = ? AND date >= ? ORDER BY date LIMIT 1""",
                    (ticker, target_date),
                ).fetchone()

                if future_row and base_price > 0:
                    ret = (future_row["close_price"] - base_price) / base_price * 100
                    returns[f"return_{label}"] = round(ret, 2)
                else:
                    returns[f"return_{label}"] = None

            enriched.append({**spike, **returns})

            # DB 저장
            conn.execute(
                """INSERT OR REPLACE INTO mention_spikes
                   (noun, spike_date, mention_count, prev_avg_count,
                    return_1d, return_7d, return_30d)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (spike["noun"], spike["spike_date"], spike["mention_count"],
                 spike["prev_avg_count"], returns.get("return_1d"),
                 returns.get("return_7d"), returns.get("return_30d")),
            )

    return enriched


def get_asset_summary(noun: str) -> Optional[Dict]:
    """특정 자산 노드의 요약 정보 (클릭 시 팝업용)."""
    if noun not in ASSET_NODES:
        return None

    with get_db() as conn:
        # 연결된 이벤트 (공동출현 Top)
        connected = conn.execute("""
            SELECT n2.noun, e.score
            FROM edges e
            JOIN nodes n1 ON (e.node_a = n1.id OR e.node_b = n1.id)
            JOIN nodes n2 ON (n2.id = CASE
                WHEN e.node_a = n1.id THEN e.node_b
                ELSE e.node_a END)
            WHERE n1.noun = ?
            ORDER BY e.score DESC LIMIT 10
        """, (noun,)).fetchall()

        # 급증일 기반 수익률
        spikes = conn.execute("""
            SELECT spike_date, mention_count, return_1d, return_7d, return_30d
            FROM mention_spikes
            WHERE noun = ?
            ORDER BY spike_date DESC
        """, (noun,)).fetchall()

        # 평균 수익률 계산
        returns_1d = [s["return_1d"] for s in spikes if s["return_1d"] is not None]
        returns_7d = [s["return_7d"] for s in spikes if s["return_7d"] is not None]
        returns_30d = [s["return_30d"] for s in spikes if s["return_30d"] is not None]

    return {
        "noun": noun,
        "asset_info": ASSET_NODES[noun],
        "connected_events": [
            {"noun": r["noun"], "score": r["score"]} for r in connected
        ],
        "spike_count": len(spikes),
        "avg_returns": {
            "1d": round(sum(returns_1d) / len(returns_1d), 2) if returns_1d else None,
            "7d": round(sum(returns_7d) / len(returns_7d), 2) if returns_7d else None,
            "30d": round(sum(returns_30d) / len(returns_30d), 2) if returns_30d else None,
        },
        "recent_spikes": [dict(s) for s in spikes[:5]],
    }


if __name__ == "__main__":
    from database import init_db
    init_db()
    print("가격 데이터 수집 중...")
    result = fetch_prices(days_back=90)
    print(result)
