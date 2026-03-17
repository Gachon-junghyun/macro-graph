"""
causal_routes.py — 인과 탐색기 전용 API 라우터
=================================================

기존 api.py의 인과 엔드포인트는 그래프 노드(27~40개)에 묶여 있어
IDF 필터로 잘려나간 2,200+ 개의 인과 노드는 접근 불가.

이 라우터는 causal_edges / causal_chains 테이블을 그래프 완전히 우회해
직접 쿼리합니다.

api.py 에 아래 두 줄 추가하면 바로 활성화됩니다:
  from causal_routes import router as causal_search_router
  app.include_router(causal_search_router)

새 엔드포인트:
  GET /api/causal/search          — 키워드 기반 인과 엣지 전문 검색
  GET /api/causal/scenario        — 시나리오: "X 가 발생하면?" 영향 트리
  GET /api/causal/chains/search   — 키워드 기반 체인 전문 검색
  GET /api/causal/explorer/stats  — 탐색기용 전체 통계
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from database import get_db

router = APIRouter()


# ══════════════════════════════════════════════════════════════
#  공통 유틸
# ══════════════════════════════════════════════════════════════

def _search_edges(q: str, category: str, min_strength: int, limit: int) -> list:
    """
    causal_edges 테이블 전문 검색.
    q 가 cause / effect / relation 중 어느 컬럼에 포함돼도 히트.
    그래프 노드 여부와 완전히 무관.
    """
    with get_db() as conn:
        params: list = []
        conditions: list = []

        if q:
            conditions.append(
                "(LOWER(cause) LIKE ? OR LOWER(effect) LIKE ? OR LOWER(relation) LIKE ?)"
            )
            kw = f"%{q.lower()}%"
            params += [kw, kw, kw]

        if category:
            conditions.append("category = ?")
            params.append(category)

        if min_strength > 1:
            conditions.append("strength >= ?")
            params.append(min_strength)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT e.id, e.cause, e.effect, e.relation, e.strength,
                   e.category, e.article_id, e.created_at,
                   a.title AS article_title
            FROM   causal_edges e
            LEFT JOIN articles a ON a.id = e.article_id
            {where}
            ORDER BY e.strength DESC, e.id DESC
            LIMIT ?
        """
        rows = conn.execute(sql, params + [limit]).fetchall()

    return [
        {
            "id":            r[0],
            "cause":         r[1],
            "effect":        r[2],
            "relation":      r[3],
            "strength":      r[4],
            "category":      r[5],
            "article_id":    r[6],
            "article_title": r[8] or "",
        }
        for r in rows
    ]


def _build_forward_tree(keyword: str, depth: int, min_strength: int) -> dict:
    """
    키워드를 포함하는 모든 cause 노드를 시작점으로
    depth 단계까지 순방향 인과 체인을 BFS로 탐색.

    Returns:
      { "roots": [노드명, ...], "tree": { 노드: [{"effect": ..., "relation": ..., ...}] } }
    """
    with get_db() as conn:

        def get_effects(cause_kw: str, exact: bool = False) -> list:
            if exact:
                sql = """
                    SELECT effect, relation, strength, category, COUNT(*) as evidence
                    FROM   causal_edges
                    WHERE  LOWER(cause) = ?
                    AND    strength >= ?
                    GROUP BY effect
                    ORDER BY evidence DESC, strength DESC
                    LIMIT 20
                """
                rows = conn.execute(sql, [cause_kw.lower(), min_strength]).fetchall()
            else:
                sql = """
                    SELECT effect, relation, strength, category, COUNT(*) as evidence
                    FROM   causal_edges
                    WHERE  LOWER(cause) LIKE ?
                    AND    strength >= ?
                    GROUP BY effect
                    ORDER BY evidence DESC, strength DESC
                    LIMIT 20
                """
                rows = conn.execute(sql, [f"%{cause_kw.lower()}%", min_strength]).fetchall()
            return [
                {"effect": r[0], "relation": r[1], "strength": r[2],
                 "category": r[3], "evidence": r[4]}
                for r in rows
            ]

        # 1단계: 키워드 포함 cause 노드 찾기
        roots_sql = """
            SELECT DISTINCT cause FROM causal_edges
            WHERE LOWER(cause) LIKE ?
            LIMIT 30
        """
        roots = [r[0] for r in conn.execute(roots_sql, [f"%{keyword.lower()}%"]).fetchall()]

        tree: dict = {}
        visited: set = set()
        queue = list(roots)

        for _ in range(depth):
            next_queue = []
            for node in queue:
                if node in visited:
                    continue
                visited.add(node)
                effects = get_effects(node, exact=True)
                if effects:
                    tree[node] = effects
                    for e in effects:
                        if e["effect"] not in visited:
                            next_queue.append(e["effect"])
            queue = next_queue

    return {"roots": roots, "tree": tree}


def _build_backward_tree(keyword: str, depth: int, min_strength: int) -> dict:
    """
    키워드를 포함하는 effect 노드를 역방향으로 depth 단계 추적.
    "무엇이 X 를 유발했나?"
    """
    with get_db() as conn:

        def get_causes(effect_kw: str, exact: bool = False) -> list:
            if exact:
                sql = """
                    SELECT cause, relation, strength, category, COUNT(*) as evidence
                    FROM   causal_edges
                    WHERE  LOWER(effect) = ?
                    AND    strength >= ?
                    GROUP BY cause
                    ORDER BY evidence DESC, strength DESC
                    LIMIT 20
                """
                rows = conn.execute(sql, [effect_kw.lower(), min_strength]).fetchall()
            else:
                sql = """
                    SELECT cause, relation, strength, category, COUNT(*) as evidence
                    FROM   causal_edges
                    WHERE  LOWER(effect) LIKE ?
                    AND    strength >= ?
                    GROUP BY cause
                    ORDER BY evidence DESC, strength DESC
                    LIMIT 20
                """
                rows = conn.execute(sql, [f"%{effect_kw.lower()}%", min_strength]).fetchall()
            return [
                {"cause": r[0], "relation": r[1], "strength": r[2],
                 "category": r[3], "evidence": r[4]}
                for r in rows
            ]

        # 1단계: 키워드 포함 effect 노드 찾기
        roots_sql = """
            SELECT DISTINCT effect FROM causal_edges
            WHERE LOWER(effect) LIKE ?
            LIMIT 30
        """
        roots = [r[0] for r in conn.execute(roots_sql, [f"%{keyword.lower()}%"]).fetchall()]

        tree: dict = {}
        visited: set = set()
        queue = list(roots)

        for _ in range(depth):
            next_queue = []
            for node in queue:
                if node in visited:
                    continue
                visited.add(node)
                causes = get_causes(node, exact=True)
                if causes:
                    tree[node] = causes
                    for c in causes:
                        if c["cause"] not in visited:
                            next_queue.append(c["cause"])
            queue = next_queue

    return {"roots": roots, "tree": tree}


# ══════════════════════════════════════════════════════════════
#  엔드포인트
# ══════════════════════════════════════════════════════════════

@router.get("/api/causal/search")
def causal_search(
    q:            str = Query("",  description="검색어 (cause / effect / relation 전문 검색)"),
    category:     str = Query("",  description="카테고리 필터 (안보 / 금융시장 / ...)"),
    min_strength: int = Query(1,   description="최소 인과 강도 (1~3)"),
    limit:        int = Query(100, description="최대 결과 수"),
):
    """
    인과 엣지 전문 검색 — 그래프 노드 필터링 완전 우회.
    cause / effect / relation 어디서든 q 가 포함되면 반환.
    """
    edges = _search_edges(q, category, min_strength, limit)
    return {
        "query":    q,
        "category": category,
        "count":    len(edges),
        "edges":    edges,
    }


@router.get("/api/causal/scenario")
def causal_scenario(
    q:            str = Query(...,  description="시나리오 키워드 (예: '전쟁', '금리 인상')"),
    depth:        int = Query(3,    description="순방향 탐색 깊이 (1~5)"),
    min_strength: int = Query(1,    description="최소 인과 강도 (1~3)"),
):
    """
    시나리오 분석: 'X 가 발생하면 무엇이 영향받나?'
    q 를 포함하는 cause 노드들로부터 depth 단계 순방향 영향 트리 반환.
    그래프 노드 여부와 무관 — DB의 모든 인과 엣지 사용.
    """
    forward  = _build_forward_tree(q, depth, min_strength)
    backward = _build_backward_tree(q, depth, min_strength)

    # 매칭 엣지 원본도 함께 반환 (상세 뷰용)
    direct_edges = _search_edges(q, "", min_strength, 200)

    return {
        "keyword":      q,
        "depth":        depth,
        "forward":      forward,   # X → ?
        "backward":     backward,  # ? → X
        "direct_edges": direct_edges,
    }


@router.get("/api/causal/chains/search")
def causal_chains_search(
    q:        str = Query("",  description="검색어 (chain_text 전문 검색)"),
    category: str = Query("",  description="카테고리 필터"),
    limit:    int = Query(50,  description="최대 결과 수"),
):
    """
    causal_chains 전문 검색.
    chain_text (A→B→C→D 형태 원문) 에서 키워드 검색.
    """
    with get_db() as conn:
        params: list = []
        conditions: list = []

        if q:
            conditions.append("LOWER(chain_text) LIKE ?")
            params.append(f"%{q.lower()}%")

        if category:
            conditions.append("category = ?")
            params.append(category)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT c.id, c.chain_text, c.category, c.confidence,
                   a.title AS article_title, a.published_at
            FROM   causal_chains c
            LEFT JOIN articles a ON a.id = c.article_id
            {where}
            ORDER BY c.confidence DESC, c.id DESC
            LIMIT ?
        """
        rows = conn.execute(sql, params + [limit]).fetchall()

    chains = [
        {
            "id":            r[0],
            "chain_text":    r[1],
            "category":      r[2],
            "confidence":    r[3],
            "article_title": r[4] or "",
            "published_at":  r[5] or "",
        }
        for r in rows
    ]
    return {"query": q, "category": category, "count": len(chains), "chains": chains}


@router.get("/api/causal/explorer/stats")
def causal_explorer_stats():
    """인과 탐색기 전용 통계 — 그래프 노드 수와 무관한 실제 데이터."""
    with get_db() as conn:
        total_edges  = conn.execute("SELECT COUNT(*) FROM causal_edges").fetchone()[0]
        total_chains = conn.execute("SELECT COUNT(*) FROM causal_chains").fetchone()[0]
        total_nodes  = conn.execute(
            "SELECT COUNT(DISTINCT noun) FROM ("
            "  SELECT cause AS noun FROM causal_edges"
            "  UNION SELECT effect AS noun FROM causal_edges"
            ")"
        ).fetchone()[0]

        categories = conn.execute(
            "SELECT category, COUNT(*) AS cnt FROM causal_edges "
            "WHERE category IS NOT NULL "
            "GROUP BY category ORDER BY cnt DESC"
        ).fetchall()

        top_causes = conn.execute(
            "SELECT cause, COUNT(*) AS cnt FROM causal_edges "
            "GROUP BY cause ORDER BY cnt DESC LIMIT 15"
        ).fetchall()

    return {
        "total_edges":  total_edges,
        "total_chains": total_chains,
        "total_nodes":  total_nodes,
        "categories":   [{"name": r[0], "count": r[1]} for r in categories],
        "top_causes":   [{"noun": r[0], "count": r[1]} for r in top_causes],
    }
