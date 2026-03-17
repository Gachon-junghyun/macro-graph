"""FastAPI 백엔드 — 그래프 데이터 API + 파이프라인 트리거."""
import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # "AIzaSyD9fumUwerjowsFrEb1Q26iatFsVnFchJU"


from database import init_db, get_db
from crawler import run_crawl, crawl_from_url_file, URL_FILE_PATH
from ingest_news_folder import ingest_news_folder, NEWS_FOLDER
from noun_extractor import process_articles, reset_concepts
from graph_builder import build_cooccurrence, compute_centrality, find_path, get_chain
from concept_normalizer import (
    self_map_all_unmapped, normalize_with_gemini,
    get_stats as get_normalizer_stats,
    get_raw_nouns_for_canonical,
)
from causal_extractor import (
    process_articles_for_chains,
    get_causal_chain, get_causal_path,
    get_impact_tree, get_category_chains,
    get_all_categories, get_top_chains,
    get_training_data_stats, export_training_data_alpaca,
)
from price_engine import (
    ASSET_NODES, fetch_prices, detect_mention_spikes,
    calculate_returns_for_spikes, get_asset_summary,
)

app = FastAPI(title="매크로 이벤트 지식 그래프", version="1.0")

# ── 인과 탐색기 라우터 (causal_routes.py) ─────────────────────
from causal_routes import router as causal_search_router
app.include_router(causal_search_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 프론트엔드 정적 파일 서빙 ──────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")


# ── 초기화 ─────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()


# ══════════════════════════════════════════════════════════════
#  그래프 데이터 API
# ══════════════════════════════════════════════════════════════

@app.get("/api/graph")
def get_graph(
    period: str = Query("3m", description="기간: 1m, 3m, all"),
    search: Optional[str] = Query(None, description="검색 키워드"),
    limit: int = Query(200, description="최대 노드 수"),
):
    """그래프 노드/엣지 데이터 반환 (D3.js 포맷)."""
    with get_db() as conn:
        # 기간 필터
        if period == "1m":
            cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        elif period == "3m":
            cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        else:
            cutoff = "2000-01-01"

        if search:
            # 검색 키워드 중심 서브그래프
            center_node = conn.execute(
                "SELECT id, noun, total_count FROM nodes WHERE noun LIKE ?",
                (f"%{search}%",),
            ).fetchone()

            if not center_node:
                return {"nodes": [], "links": [], "message": f"'{search}' 노드를 찾을 수 없습니다"}

            # 중심 노드 + 연결 노드
            edges = conn.execute("""
                SELECT e.node_a, e.node_b, e.score, e.article_count
                FROM edges e
                WHERE (e.node_a = ? OR e.node_b = ?) AND e.last_seen >= ?
                ORDER BY e.score DESC
                LIMIT ?
            """, (center_node["id"], center_node["id"], cutoff, limit)).fetchall()

            node_ids = {center_node["id"]}
            for e in edges:
                node_ids.add(e["node_a"])
                node_ids.add(e["node_b"])

            nodes = conn.execute(
                f"SELECT id, noun, total_count FROM nodes WHERE id IN ({','.join('?' * len(node_ids))})",
                list(node_ids),
            ).fetchall()

        else:
            # 전체 그래프 (상위 노드 기준)
            nodes = conn.execute("""
                SELECT id, noun, total_count FROM nodes
                WHERE last_seen >= ?
                ORDER BY total_count DESC LIMIT ?
            """, (cutoff, limit)).fetchall()

            node_ids = {n["id"] for n in nodes}
            placeholders = ",".join("?" * len(node_ids))

            edges = conn.execute(f"""
                SELECT node_a, node_b, score, article_count
                FROM edges
                WHERE node_a IN ({placeholders}) AND node_b IN ({placeholders})
                  AND last_seen >= ?
                ORDER BY score DESC
                LIMIT 1000
            """, list(node_ids) + list(node_ids) + [cutoff]).fetchall()

        # D3.js 포맷 변환
        d3_nodes = []
        for n in nodes:
            is_asset = n["noun"] in ASSET_NODES
            d3_nodes.append({
                "id": n["id"],
                "noun": n["noun"],
                "count": n["total_count"],
                "isAsset": is_asset,
                "assetType": ASSET_NODES[n["noun"]]["type"] if is_asset else None,
            })

        d3_links = [
            {
                "source": e["node_a"],
                "target": e["node_b"],
                "score": e["score"],
                "articleCount": e["article_count"],
            }
            for e in edges
        ]

    return {
        "nodes": d3_nodes,
        "links": d3_links,
        "meta": {
            "period": period,
            "search": search,
            "nodeCount": len(d3_nodes),
            "linkCount": len(d3_links),
        },
    }


@app.get("/api/node/{node_id}")
def get_node_detail(node_id: int):
    """노드 클릭 시 상세 정보."""
    with get_db() as conn:
        node = conn.execute(
            "SELECT id, noun, total_count, last_seen FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()

        if not node:
            return {"error": "노드를 찾을 수 없습니다"}

        # 연결된 상위 10개 노드
        connected = conn.execute("""
            SELECT n2.id, n2.noun, n2.total_count, e.score, e.article_count
            FROM edges e
            JOIN nodes n2 ON n2.id = CASE
                WHEN e.node_a = ? THEN e.node_b ELSE e.node_a END
            WHERE e.node_a = ? OR e.node_b = ?
            ORDER BY e.score DESC LIMIT 10
        """, (node_id, node_id, node_id)).fetchall()

        # 최근 관련 기사 — canonical + 모든 alias 기준으로 검색
        all_nouns = get_raw_nouns_for_canonical(node["noun"])
        placeholders = ",".join("?" * len(all_nouns))
        recent_articles = conn.execute(f"""
            SELECT DISTINCT a.id, a.title, a.source, a.published_at, a.url
            FROM article_nouns an
            JOIN articles a ON an.article_id = a.id
            WHERE an.noun IN ({placeholders})
            ORDER BY a.published_at DESC LIMIT 10
        """, all_nouns).fetchall()

        result = {
            "node": dict(node),
            "connected": [dict(c) for c in connected],
            "recentArticles": [dict(a) for a in recent_articles],
            "isAsset": node["noun"] in ASSET_NODES,
        }

        # 자산 노드면 가격 데이터도 추가
        if node["noun"] in ASSET_NODES:
            asset_summary = get_asset_summary(node["noun"])
            if asset_summary:
                result["assetSummary"] = asset_summary

    return result


@app.get("/api/centrality")
def get_centrality(top_n: int = Query(30)):
    """중심성 높은 노드 순위."""
    results = compute_centrality()
    return {"centrality": results[:top_n]}


@app.get("/api/asset/{noun}")
def get_asset_detail(noun: str):
    """자산 노드 상세 (가격 + 수익률)."""
    summary = get_asset_summary(noun)
    if not summary:
        return {"error": f"'{noun}'은 자산 노드가 아닙니다"}

    # 가격 히스토리
    with get_db() as conn:
        ticker = ASSET_NODES[noun]["ticker"]
        prices = conn.execute("""
            SELECT date, close_price FROM price_data
            WHERE ticker = ? ORDER BY date
        """, (ticker,)).fetchall()

        summary["priceHistory"] = [
            {"date": p["date"], "price": p["close_price"]} for p in prices
        ]

    return summary


@app.get("/api/stats")
def get_stats():
    """전체 통계."""
    with get_db() as conn:
        article_count = conn.execute("SELECT COUNT(*) as c FROM articles").fetchone()["c"]
        node_count = conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
        edge_count = conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]
        noun_count = conn.execute("SELECT COUNT(DISTINCT noun) as c FROM article_nouns").fetchone()["c"]
        latest = conn.execute(
            "SELECT MAX(published_at) as latest FROM articles"
        ).fetchone()["latest"]

    return {
        "articles": article_count,
        "nodes": node_count,
        "edges": edge_count,
        "uniqueNouns": noun_count,
        "latestArticle": latest,
        "assetNodes": len(ASSET_NODES),
    }


# ── 공동출현 그래프 경로 탐색 ─────────────────────────────────

@app.get("/api/path")
def get_path(
    node_a: str = Query(..., description="시작 노드"),
    node_b: str = Query(..., description="끝 노드"),
    max_depth: int = Query(5, description="최대 홉 수"),
):
    """두 노드 사이 최단 경로 (공동출현 그래프 기반)."""
    return find_path(node_a, node_b, max_depth=max_depth)


@app.get("/api/chain/{noun}")
def get_node_chain(
    noun: str,
    depth: int = Query(2, description="탐색 깊이 (1~3 권장)"),
    top_k: int = Query(5, description="각 레벨 상위 K개"),
):
    """특정 노드에서 depth 홉까지 연결된 체인 (공동출현)."""
    return get_chain(noun, depth=depth, top_k=top_k)


# ── 인과 관계 ─────────────────────────────────────────────────

@app.get("/api/causal/chain/{noun}")
def get_causal_chain_api(
    noun: str,
    depth: int = Query(3, description="탐색 깊이"),
    direction: str = Query("both", description="forward|backward|both"),
):
    """인과 그래프에서 특정 노드의 원인/결과 체인."""
    return get_causal_chain(noun, depth=depth, direction=direction)


@app.get("/api/causal/path")
def get_causal_path_api(
    node_a: str = Query(...),
    node_b: str = Query(...),
    max_depth: int = Query(5),
):
    """인과 그래프에서 A→B 인과 경로."""
    return get_causal_path(node_a, node_b, max_depth=max_depth)


@app.get("/api/causal/impact/{noun}")
def get_impact_api(
    noun: str,
    depth: int = Query(4, description="영향 탐색 깊이 (1~5 권장)"),
):
    """
    '전쟁이 일어나면 무엇이 영향받나?' 형태의 순방향 영향 트리.
    noun을 트리거로 downstream 효과를 depth 단계까지 추적.
    evidence(근거 기사 수)와 strength(인과 강도)로 정렬.
    """
    return get_impact_tree(noun, depth=depth)


@app.get("/api/causal/categories")
def get_categories_api():
    """카테고리별 체인 수 통계 (실물경제, 지정학, 에너지시장 등)."""
    return {"categories": get_all_categories()}


@app.get("/api/causal/chains/{category}")
def get_chains_by_category_api(
    category: str,
    limit: int = Query(20, description="반환 최대 체인 수"),
):
    """특정 카테고리의 대표 체인 목록 (증거 기사 수 순)."""
    return {
        "category": category,
        "chains": get_category_chains(category, limit=limit),
    }


@app.get("/api/causal/top-chains")
def get_top_chains_api(limit: int = Query(30)):
    """전체 카테고리에서 증거가 많은 상위 체인 목록."""
    return {"chains": get_top_chains(limit=limit)}


# ── 학습 데이터 ────────────────────────────────────────────────

@app.get("/api/training/stats")
def get_training_stats():
    """저장된 학습 데이터 통계 (건수, 파일 크기)."""
    return get_training_data_stats()


@app.post("/api/training/export-alpaca")
def export_alpaca():
    """ChatML → Alpaca 포맷으로 변환 내보내기."""
    path = export_training_data_alpaca()
    return {"exported_to": path}


@app.get("/api/explain")
async def explain_relation(
    node_a: str = Query(..., description="첫 번째 노드 명사"),
    node_b: str = Query(..., description="두 번째 노드 명사"),
):
    """
    두 노드가 왜 연결됐는지 Gemini AI로 설명.
    두 명사가 함께 등장한 기사들을 근거로 상관관계를 분석해 반환.
    GEMINI_API_KEY 없으면 기사 목록만 반환.
    """
    # 두 명사가 함께 등장한 기사 최대 5건 가져오기
    with get_db() as conn:
        rows = conn.execute("""
            SELECT DISTINCT a.id, a.title, a.body, a.source, a.published_at
            FROM articles a
            JOIN article_nouns an1 ON an1.article_id = a.id AND an1.noun = ?
            JOIN article_nouns an2 ON an2.article_id = a.id AND an2.noun = ?
            ORDER BY a.published_at DESC
            LIMIT 5
        """, (node_a, node_b)).fetchall()

        # 엣지 점수도 가져오기
        edge = conn.execute("""
            SELECT e.score, e.article_count
            FROM edges e
            JOIN nodes na ON na.id = e.node_a AND na.noun = ?
            JOIN nodes nb ON nb.id = e.node_b AND nb.noun = ?
            UNION
            SELECT e.score, e.article_count
            FROM edges e
            JOIN nodes na ON na.id = e.node_b AND na.noun = ?
            JOIN nodes nb ON nb.id = e.node_a AND nb.noun = ?
            LIMIT 1
        """, (node_a, node_b, node_a, node_b)).fetchone()

    articles = [dict(r) for r in rows]
    edge_info = dict(edge) if edge else {}

    if not articles:
        return {
            "node_a": node_a,
            "node_b": node_b,
            "explanation": f"'{node_a}'와 '{node_b}'가 함께 등장한 기사가 없습니다.",
            "articles": [],
            "edge": edge_info,
        }

    if not GEMINI_API_KEY:
        return {
            "node_a": node_a,
            "node_b": node_b,
            "explanation": "GEMINI_API_KEY가 설정되지 않았습니다. 환경변수를 설정하세요.",
            "articles": [{"title": a["title"], "source": a["source"], "date": a["published_at"]} for a in articles],
            "edge": edge_info,
        }

    # 기사 본문 요약 컨텍스트 구성
    context_parts = []
    for i, a in enumerate(articles, 1):
        body_preview = (a["body"] or "")[:600]
        context_parts.append(f"[기사{i}] {a['title']}\n{body_preview}")
    context = "\n\n".join(context_parts)

    prompt = f"""다음 뉴스 기사들을 분석하여 '{node_a}'와 '{node_b}' 사이의 상관관계를 설명해주세요.

기사 (공동 출현 {edge_info.get('article_count', len(articles))}건, 연결 강도 {edge_info.get('score', '-')}점):
{context}

아래 형식으로 3~5문장 이내로 간결하게 답변해주세요:
- 두 키워드가 어떤 맥락에서 함께 등장하는지
- 인과관계 또는 연쇄 관계가 있다면 어떤 방향인지
- 투자/시장 관점에서 시사점이 있다면 한 줄로

전문 용어는 그대로 사용하고, 불필요한 수식어 없이 핵심만 서술하세요."""

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
        )
        explanation = response.text.strip()
    except Exception as e:
        explanation = f"AI 분석 오류: {e}"

    return {
        "node_a": node_a,
        "node_b": node_b,
        "explanation": explanation,
        "articles": [{"title": a["title"], "source": a["source"], "date": a["published_at"]} for a in articles],
        "edge": edge_info,
    }


# ══════════════════════════════════════════════════════════════
#  파이프라인 트리거 API
# ══════════════════════════════════════════════════════════════

@app.post("/api/pipeline/crawl")
def trigger_crawl():
    """크롤링 실행."""
    result = run_crawl()
    return result


@app.post("/api/pipeline/crawl-urls")
def trigger_crawl_urls():
    """
    urls.txt 파일에 적힌 URL들을 크롤링해서 DB에 저장.

    - 파일 내 중복 URL 자동 제거
    - DB에 이미 있는 URL 자동 건너뜀
    - 결과에 처리 통계 포함
    """
    result = crawl_from_url_file(URL_FILE_PATH)
    return result


@app.get("/api/pipeline/crawl-urls/preview")
def preview_url_file():
    """
    urls.txt 파일 내용을 미리 보기.
    실제 크롤링 없이 URL 목록과 중복 여부만 반환.
    """
    import os
    if not os.path.exists(URL_FILE_PATH):
        return {"error": f"파일 없음: {URL_FILE_PATH}"}

    raw_urls = []
    with open(URL_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            url = line.split("#")[0].strip()
            if url:
                raw_urls.append(url)

    seen = set()
    unique_urls = []
    for u in raw_urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    # DB 중복 확인
    from database import get_db
    db_results = []
    with get_db() as conn:
        for url in unique_urls:
            row = conn.execute(
                "SELECT id, title FROM articles WHERE url = ?", (url,)
            ).fetchone()
            db_results.append({
                "url": url,
                "in_db": row is not None,
                "db_title": row["title"] if row else None,
            })

    return {
        "file_path":       URL_FILE_PATH,
        "total_in_file":   len(raw_urls),
        "file_duplicates": len(raw_urls) - len(unique_urls),
        "unique_urls":     len(unique_urls),
        "urls":            db_results,
    }


@app.post("/api/pipeline/ingest-news")
def trigger_ingest_news():
    """
    news/ 폴더 안의 .txt 파일을 읽어 DB에 저장.
    파일 형식: 제목:/날짜:(선택)/---/본문
    """
    result = ingest_news_folder(NEWS_FOLDER)
    return result


@app.post("/api/pipeline/extract")
def trigger_extract(batch_size: int = Query(50)):
    """Gemini로 기사에서 핵심 개념 추출 (v2 — kiwipiepy 제거)."""
    result = process_articles(batch_size=batch_size)
    return result


@app.get("/api/pipeline/normalize/stats")
def trigger_normalize_stats():
    """정규화 현황 통계 (raw noun 수, canonical 수, 통합률 등)."""
    return get_normalizer_stats()


@app.post("/api/pipeline/normalize/self-map")
def trigger_self_map():
    """
    Gemini 없이 즉시 실행 — 기존 article_nouns의 모든 개념을 1:1 자기 매핑.
    graph_builder 실행 전에 먼저 호출하면 기존 데이터 호환성 즉시 확보.
    """
    result = self_map_all_unmapped()
    return result


@app.post("/api/pipeline/normalize/gemini")
def trigger_gemini_normalize(
    batch_size: int = Query(60, description="배치당 개념 수"),
    rate_limit_sec: float = Query(6.0, description="배치 간 대기(초)"),
    max_batches: int = Query(10, description="최대 배치 수 (테스트용)"),
):
    """
    Gemini로 유사 개념 통합 — '트럼프 관세' / '트럼프 관세 부과' → 하나의 노드.
    실행 후 /pipeline/build-graph 재실행 필요.
    """
    result = normalize_with_gemini(
        batch_size=batch_size,
        rate_limit_sec=rate_limit_sec,
        max_batches=max_batches,
    )
    return result


@app.post("/api/pipeline/reset-concepts")
def trigger_reset_concepts():
    """
    기존 article_nouns / nodes / edges 전부 초기화.
    kiwipiepy 기반 저품질 노드를 날리고 Gemini 추출로 재시작할 때 사용.
    ⚠️ 되돌릴 수 없음. 이후 /extract → /build-graph 순서로 재처리 필요.
    """
    return reset_concepts()


@app.post("/api/pipeline/build-graph")
def trigger_build_graph(days: int = Query(90)):
    """그래프 빌드 실행."""
    result = build_cooccurrence(days_back=days)
    return result


@app.post("/api/pipeline/fetch-prices")
def trigger_fetch_prices(days: int = Query(90)):
    """가격 데이터 수집."""
    result = fetch_prices(days_back=days)
    return result


@app.post("/api/pipeline/detect-spikes")
def trigger_detect_spikes():
    """언급 급증 감지 + 수익률 계산."""
    spikes = detect_mention_spikes()
    enriched = calculate_returns_for_spikes(spikes)
    return {"spikes": enriched, "count": len(enriched)}


@app.post("/api/pipeline/extract-causality")
def trigger_extract_causality(
    batch_size: int = Query(30, description="한 번에 처리할 기사 수"),
    rate_limit_sec: float = Query(7.0, description="호출 간 대기(초). 무료=7, 유료=0.5"),
):
    """
    Gemini로 기사에서 인과 체인 추출 (GEMINI_API_KEY 필요).
    체인 노드를 article_nouns에 자동 저장 → noun_extractor 별도 실행 불필요.
    무료 플랜: rate_limit_sec=7 (기본값) / 유료 플랜: rate_limit_sec=0.5
    """
    return process_articles_for_chains(batch_size=batch_size, rate_limit_sec=rate_limit_sec)


@app.post("/api/pipeline/full")
def trigger_full_pipeline():
    """전체 파이프라인 실행 (1→2→3→4단계)."""
    results = {}

    print("\n[1/5] 크롤링...")
    results["crawl"] = run_crawl()

    print("[2/5] 명사 추출...")
    results["extract"] = process_articles()

    print("[3/5] 그래프 빌드...")
    results["graph"] = build_cooccurrence()

    print("[4/5] 가격 수집...")
    try:
        results["prices"] = fetch_prices(days_back=90)
    except Exception as e:
        results["prices"] = {"error": str(e)}

    print("[5/5] 인과 체인 추출 (Gemini)...")
    try:
        results["causal_chains"] = process_articles_for_chains(batch_size=50)
    except Exception as e:
        results["causal_chains"] = {"error": str(e)}

    print("[6/6] 급증 감지...")
    try:
        spikes = detect_mention_spikes()
        results["spikes"] = calculate_returns_for_spikes(spikes)
    except Exception as e:
        results["spikes"] = {"error": str(e)}

    return results


# ── 프론트엔드 서빙 ────────────────────────────────────────────
if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
