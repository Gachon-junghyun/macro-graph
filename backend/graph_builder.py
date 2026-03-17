"""
공동출현 그래프 생성 — 개념 쌍 간 엣지 & 스코어 계산.

입력: article_nouns 테이블 (Gemini 추출 개념)
출력: nodes / edges 테이블 (방향 없는 공동출현 그래프)

스코어 기준:
  같은 문장 공동출현: 3점
  같은 기사 공동출현: 2점
  같은 날 다른 기사:  1점
"""
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set

import networkx as nx

from database import get_db
from concept_normalizer import load_alias_map


def _split_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 분리 (정규식 기반)."""
    if not text:
        return []
    sentences = re.split(r"[.!?\n]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def build_cooccurrence(days_back: int = 90) -> Dict:
    """공동출현 그래프를 DB에서 구축.

    스코어 계산:
    - 같은 문장 등장: 3점
    - 같은 기사 등장: 2점
    - 같은 날 다른 기사: 1점
    """
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # ── 정규화 alias 맵 로드 ──────────────────────────────────────
    # {raw_noun: canonical} — 없으면 빈 dict (기존 동작 유지)
    alias_map = load_alias_map()
    # 역방향 맵: {canonical: [raw_nouns]} — 문장 내 alias 검색에 사용
    reverse_alias_map: Dict[str, List[str]] = {}
    for raw, canon in alias_map.items():
        reverse_alias_map.setdefault(canon, []).append(raw)

    if alias_map:
        print(f"  정규화 매핑 로드: {len(alias_map)}개 raw → {len(reverse_alias_map)}개 canonical")
    else:
        print("  정규화 매핑 없음 — raw noun 그대로 사용 (self_map_all_unmapped() 권장)")

    # 엣지 스코어 누적 dict: (noun_a, noun_b) -> score
    edge_scores: Dict[Tuple[str, str], float] = defaultdict(float)
    # 엣지별 기사 수
    edge_article_count: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    # 노드 등장 빈도
    node_counts: Dict[str, int] = defaultdict(int)
    # 날짜별-명사별 기사ID
    date_noun_articles: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))

    with get_db() as conn:
        # 모든 기사 + 명사 로드
        articles = conn.execute("""
            SELECT a.id, a.title, a.body, a.published_at
            FROM articles a
            WHERE a.published_at >= ?
            ORDER BY a.published_at
        """, (cutoff,)).fetchall()

        print(f"그래프 빌드: {len(articles)}개 기사 처리 시작")

        for art in articles:
            art_id = art["id"]
            title = art["title"] or ""
            body = art["body"] or ""
            pub_date = (art["published_at"] or "")[:10]

            # 이 기사의 명사 로드
            rows = conn.execute(
                "SELECT noun, position FROM article_nouns WHERE article_id = ?",
                (art_id,),
            ).fetchall()

            article_nouns: Set[str] = set()
            title_nouns: Set[str] = set()

            for r in rows:
                noun = r["noun"]
                if noun.startswith("__"):   # __processed__ 등 내부 마커 제외
                    continue
                # ── 정규화: raw_noun → canonical (없으면 raw 그대로) ──
                canonical = alias_map.get(noun, noun)
                article_nouns.add(canonical)
                node_counts[canonical] = node_counts.get(canonical, 0) + 1
                if r["position"] == "title":
                    title_nouns.add(canonical)

                # 날짜별 명사-기사 매핑
                if pub_date:
                    date_noun_articles[pub_date][canonical].add(art_id)

            # ── 같은 기사 공동출현 (2점) ──
            noun_list = sorted(article_nouns)
            for i in range(len(noun_list)):
                for j in range(i + 1, len(noun_list)):
                    pair = (noun_list[i], noun_list[j])
                    edge_scores[pair] += 2
                    edge_article_count[pair].add(art_id)

            # ── 같은 문장 공동출현 (추가 1점 = 총 3점) ──
            all_text = title + ". " + body
            sentences = _split_sentences(all_text)

            for sent in sentences:
                # 이 문장에 포함된 canonical 찾기
                # canonical 직접 검색 OR canonical의 raw alias가 문장에 있으면 매칭
                sent_nouns = []
                for canonical in article_nouns:
                    if canonical in sent:
                        sent_nouns.append(canonical)
                    elif reverse_alias_map:
                        raw_list = reverse_alias_map.get(canonical, [])
                        if any(raw in sent for raw in raw_list):
                            sent_nouns.append(canonical)
                sent_nouns = sorted(set(sent_nouns))
                for i in range(len(sent_nouns)):
                    for j in range(i + 1, len(sent_nouns)):
                        pair = (sent_nouns[i], sent_nouns[j])
                        edge_scores[pair] += 1  # 추가 1점 (이미 기사 레벨에서 2점)

        # ── 같은 날 다른 기사 (1점) ──
        for date_str, noun_arts in date_noun_articles.items():
            nouns_today = sorted(noun_arts.keys())
            for i in range(len(nouns_today)):
                for j in range(i + 1, len(nouns_today)):
                    na, nb = nouns_today[i], nouns_today[j]
                    arts_a = noun_arts[na]
                    arts_b = noun_arts[nb]
                    # 다른 기사에서 등장한 경우만
                    cross_articles = (arts_a - arts_b) | (arts_b - arts_a)
                    if cross_articles:
                        pair = (na, nb)
                        edge_scores[pair] += 1
                        edge_article_count[pair].update(arts_a | arts_b)

    # ── 노이즈 제거 ──────────────────────────────────────────
    total_articles = len(articles) if articles else 1

    # 실제 개념이 추출된 기사 수 — DB에서 직접 조회
    # (전체 기사 수로 계산하면 처리율이 낮을 때 필터가 과도하게 엄격해짐)
    with get_db() as conn:
        processed_articles = conn.execute("""
            SELECT COUNT(DISTINCT article_id) as c
            FROM article_nouns
            WHERE noun NOT GLOB '__*'
        """).fetchone()["c"]
    processed_articles = max(processed_articles, 1)

    # 1) 데이터 밀도별 적응형 필터
    #    처리율이 낮을수록 느슨하게, 쌓일수록 엄격하게 자동 조정
    if processed_articles < 100:
        # 초기 단계: 최소 2개 기사 등장, 엣지는 1개 기사면 충분
        min_node_count = 2
        min_articles_for_edge = 1
    elif processed_articles < 500:
        # 중간 단계
        min_node_count = max(2, int(processed_articles * 0.015))
        min_articles_for_edge = max(1, processed_articles // 50)
    else:
        # 데이터 충분: 기존 엄격한 기준 적용
        min_node_count = max(2, int(processed_articles * 0.01))
        min_articles_for_edge = max(1, min(10, processed_articles // 20))

    print(f"  필터 기준: 전체 {total_articles}개 / 처리된 {processed_articles}개 기사 "
          f"→ min_node_count={min_node_count}, min_edge_articles={min_articles_for_edge}")

    # 2) IDF 기반 상한 필터 — 처리된 기사의 60% 이상 등장하면 노이즈
    MAX_IDF_RATIO = 0.60
    valid_nodes = {
        n for n, c in node_counts.items()
        if min_node_count <= c <= int(processed_articles * MAX_IDF_RATIO)
    }
    valid_edges = {}
    for pair, score in edge_scores.items():
        na, nb = pair
        if na not in valid_nodes or nb not in valid_nodes:
            continue
        art_count = len(edge_article_count.get(pair, set()))
        if art_count < min_articles_for_edge:
            continue
        valid_edges[pair] = {"score": score, "article_count": art_count}

    # 하위 20% 제거
    if valid_edges:
        scores = sorted(v["score"] for v in valid_edges.values())
        threshold = scores[int(len(scores) * 0.2)]
        valid_edges = {
            k: v for k, v in valid_edges.items() if v["score"] >= threshold
        }

    # ── DB에 저장 ──
    with get_db() as conn:
        # 기존 데이터 클리어
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM nodes")

        # 노드 저장
        node_id_map = {}
        for noun in valid_nodes:
            if node_counts.get(noun, 0) >= min_node_count:
                conn.execute(
                    "INSERT OR REPLACE INTO nodes (noun, total_count, last_seen) VALUES (?, ?, ?)",
                    (noun, node_counts[noun], datetime.now().strftime("%Y-%m-%d")),
                )
                row = conn.execute(
                    "SELECT id FROM nodes WHERE noun = ?", (noun,)
                ).fetchone()
                node_id_map[noun] = row["id"]

        # 엣지 저장
        edge_count = 0
        for (na, nb), info in valid_edges.items():
            if na in node_id_map and nb in node_id_map:
                conn.execute(
                    """INSERT OR REPLACE INTO edges
                       (node_a, node_b, score, article_count, last_seen)
                       VALUES (?, ?, ?, ?, ?)""",
                    (node_id_map[na], node_id_map[nb],
                     info["score"], info["article_count"],
                     datetime.now().strftime("%Y-%m-%d")),
                )
                edge_count += 1

    result = {
        "total_nodes": len(node_id_map),
        "total_edges": edge_count,
        "removed_nodes": len(node_counts) - len(valid_nodes),
        "removed_edges": len(edge_scores) - len(valid_edges),
    }
    print(f"그래프 빌드 완료: {result}")
    return result


def get_networkx_graph() -> nx.Graph:
    """DB에서 NetworkX 그래프 객체 생성."""
    G = nx.Graph()

    with get_db() as conn:
        nodes = conn.execute("SELECT id, noun, total_count FROM nodes").fetchall()
        for n in nodes:
            G.add_node(n["id"], noun=n["noun"], count=n["total_count"])

        edges = conn.execute(
            "SELECT node_a, node_b, score, article_count FROM edges"
        ).fetchall()
        for e in edges:
            G.add_edge(
                e["node_a"], e["node_b"],
                score=e["score"], article_count=e["article_count"],
            )

    return G


def compute_centrality() -> List[Dict]:
    """중심성 높은 노드 계산."""
    G = get_networkx_graph()
    if len(G.nodes) == 0:
        return []

    # 다양한 중심성 지표 계산
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    results = []
    for node_id in G.nodes:
        results.append({
            "node_id": node_id,
            "noun": G.nodes[node_id]["noun"],
            "count": G.nodes[node_id]["count"],
            "degree_centrality": round(degree_cent.get(node_id, 0), 4),
            "betweenness": round(betweenness.get(node_id, 0), 4),
        })

    results.sort(key=lambda x: x["degree_centrality"], reverse=True)
    return results


def find_path(noun_a: str, noun_b: str, max_depth: int = 5) -> Dict:
    """
    두 노드 사이의 최단 경로(인과 체인) 탐색.
    예: 후티 → 홍해 → 유가 → 인플레
    """
    G = get_networkx_graph()

    # 명사 → node_id 매핑
    noun_to_id = {data["noun"]: nid for nid, data in G.nodes(data=True)}
    id_to_noun = {nid: data["noun"] for nid, data in G.nodes(data=True)}

    id_a = noun_to_id.get(noun_a)
    id_b = noun_to_id.get(noun_b)

    if id_a is None:
        return {"error": f"'{noun_a}' 노드를 찾을 수 없습니다"}
    if id_b is None:
        return {"error": f"'{noun_b}' 노드를 찾을 수 없습니다"}
    if id_a == id_b:
        return {"path": [noun_a], "length": 0, "hops": []}

    try:
        path_ids = nx.shortest_path(G, id_a, id_b)
    except nx.NetworkXNoPath:
        return {"error": f"'{noun_a}'와 '{noun_b}' 사이에 경로가 없습니다 (연결 안 된 컴포넌트)"}

    if len(path_ids) - 1 > max_depth:
        return {"error": f"경로가 너무 깁니다 ({len(path_ids)-1}홉, 최대 {max_depth}홉)"}

    # 경로의 각 엣지 정보 포함
    hops = []
    for i in range(len(path_ids) - 1):
        a, b = path_ids[i], path_ids[i + 1]
        edge_data = G.edges[a, b]
        hops.append({
            "from": id_to_noun[a],
            "to":   id_to_noun[b],
            "score": edge_data.get("score", 0),
            "article_count": edge_data.get("article_count", 0),
        })

    return {
        "path":   [id_to_noun[nid] for nid in path_ids],
        "length": len(path_ids) - 1,
        "hops":   hops,
    }


def get_chain(noun: str, depth: int = 2, top_k: int = 5) -> Dict:
    """
    특정 노드에서 depth 홉까지 연결된 노드들을 탐색.
    각 레벨에서 엣지 스코어 상위 top_k개만 유지.
    """
    G = get_networkx_graph()
    noun_to_id = {data["noun"]: nid for nid, data in G.nodes(data=True)}
    id_to_noun = {nid: data["noun"] for nid, data in G.nodes(data=True)}

    start_id = noun_to_id.get(noun)
    if start_id is None:
        return {"error": f"'{noun}' 노드를 찾을 수 없습니다"}

    visited = {start_id}
    levels = []
    current_level = {start_id}

    for d in range(depth):
        next_level_candidates = []
        for node_id in current_level:
            neighbors = list(G.neighbors(node_id))
            for nb in neighbors:
                if nb in visited:
                    continue
                edge_data = G.edges[node_id, nb]
                next_level_candidates.append({
                    "parent": id_to_noun[node_id],
                    "node":   id_to_noun[nb],
                    "node_id": nb,
                    "score":  edge_data.get("score", 0),
                    "article_count": edge_data.get("article_count", 0),
                })

        # 스코어 상위 top_k만 유지
        next_level_candidates.sort(key=lambda x: x["score"], reverse=True)
        kept = next_level_candidates[:top_k * len(current_level)]

        level_nodes = []
        next_level = set()
        for c in kept:
            if c["node_id"] not in visited:
                visited.add(c["node_id"])
                next_level.add(c["node_id"])
                level_nodes.append({
                    "parent": c["parent"],
                    "noun":   c["node"],
                    "score":  c["score"],
                    "article_count": c["article_count"],
                    "depth":  d + 1,
                })

        levels.append(level_nodes)
        current_level = next_level
        if not current_level:
            break

    return {
        "root":   noun,
        "depth":  depth,
        "levels": levels,
        "total_nodes": sum(len(l) for l in levels),
    }


if __name__ == "__main__":
    from database import init_db
    init_db()
    result = build_cooccurrence()
    print("\n중심성 Top 20:")
    for item in compute_centrality()[:20]:
        print(f"  {item['noun']}: degree={item['degree_centrality']}, "
              f"betweenness={item['betweenness']}, count={item['count']}")

    print("\n경로 탐색 테스트:")
    print(find_path("후티", "인플레"))
    print("\n체인 탐색 테스트:")
    chain = get_chain("삼성전자", depth=2)
    for lv, nodes in enumerate(chain.get("levels", []), 1):
        print(f"  Depth {lv}:", [n["noun"] for n in nodes[:5]])
