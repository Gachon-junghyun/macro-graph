"""
macro-graph 기사 검색 앱
실행: streamlit run search_app.py
"""

import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

# ── 설정 ──────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "macro_graph.db"

SOURCE_MAP = {
    "yonhap":    "연합뉴스",
    "hankyung":  "한국경제",
    "mk":        "매일경제",
    "einfomax":  "연합인포맥스",
}
SOURCE_COLOR = {
    "연합뉴스":    "#e74c3c",
    "한국경제":    "#2980b9",
    "매일경제":    "#27ae60",
    "연합인포맥스": "#8e44ad",
}

st.set_page_config(
    page_title="macro-graph 기사 검색",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 전역 스타일 ────────────────────────────────────────────────
st.markdown("""
<style>
/* 카드 */
.article-card {
    background: #1e1e2e;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.article-card:hover { border-color: #5865f2; }

/* 소스 배지 */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
    color: white;
    margin-right: 8px;
}

/* 제목 */
.art-title {
    font-size: 15px;
    font-weight: 600;
    color: #e0e0e0;
    line-height: 1.5;
    margin: 6px 0 4px 0;
}

/* 날짜 */
.art-date {
    font-size: 12px;
    color: #888;
    margin-bottom: 6px;
}

/* 본문 미리보기 */
.art-body {
    font-size: 13px;
    color: #aaa;
    line-height: 1.6;
    border-left: 3px solid #2d2d44;
    padding-left: 10px;
    margin-top: 8px;
}

/* 링크 버튼 */
.art-link {
    display: inline-block;
    margin-top: 8px;
    font-size: 12px;
    color: #5865f2;
    text-decoration: none;
}
.art-link:hover { color: #7b8cf8; text-decoration: underline; }

/* 검색 결과 수 */
.result-count {
    font-size: 14px;
    color: #aaa;
    margin-bottom: 16px;
}

/* 구분선 */
hr { border-color: #2d2d44; }
</style>
""", unsafe_allow_html=True)


# ── DB 연결 (캐시) ─────────────────────────────────────────────
@st.cache_resource
def get_connection():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


@st.cache_data(ttl=60)
def get_date_range():
    conn = get_connection()
    row = conn.execute(
        "SELECT MIN(published_at), MAX(published_at) FROM articles"
    ).fetchone()
    return row[0], row[1]


@st.cache_data(ttl=30, show_spinner=False)
def search_articles(keyword, date_from, date_to, sources, limit=200):
    conn = get_connection()

    conditions = ["1=1"]
    params = []

    if keyword.strip():
        kw = f"%{keyword.strip()}%"
        conditions.append("(title LIKE ? OR body LIKE ?)")
        params += [kw, kw]

    if date_from:
        conditions.append("published_at >= ?")
        params.append(str(date_from))

    if date_to:
        conditions.append("published_at <= ?")
        params.append(str(date_to))

    if sources:
        ph = ",".join("?" * len(sources))
        conditions.append(f"source IN ({ph})")
        params += list(sources)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, source, title, body, url, published_at
        FROM articles
        WHERE {where}
        ORDER BY published_at DESC, id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(sql, conn, params=params)
    return df


# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 검색 조건")
    st.markdown("---")

    keyword = st.text_input(
        "키워드",
        placeholder="예: 금리, 환율, 삼성, AI ...",
        help="기사 제목과 본문 전체를 검색합니다",
    )

    st.markdown("### 📅 날짜 범위")
    min_date, max_date = get_date_range()

    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("시작일", value=None, min_value=None, max_value=None, key="df")
    with col2:
        date_to   = st.date_input("종료일", value=None, min_value=None, max_value=None, key="dt")

    st.markdown(f"<div style='font-size:11px;color:#666;margin-top:-8px'>DB 기간: {min_date} ~ {max_date}</div>", unsafe_allow_html=True)

    st.markdown("### 📰 출처")
    source_checks = {}
    for src_key, src_label in SOURCE_MAP.items():
        source_checks[src_key] = st.checkbox(src_label, value=True, key=f"src_{src_key}")

    st.markdown("### ⚙️ 결과 수")
    result_limit = st.select_slider(
        "최대 결과",
        options=[50, 100, 200, 500],
        value=200,
    )

    st.markdown("---")
    search_btn = st.button("🔍 검색", type="primary", use_container_width=True)

    # 통계 요약
    st.markdown("---")
    st.markdown("### 📊 DB 현황")
    conn = get_connection()
    stats = conn.execute(
        "SELECT source, COUNT(*) FROM articles GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall()
    for src, cnt in stats:
        label = SOURCE_MAP.get(src, src)
        color = SOURCE_COLOR.get(label, "#999")
        st.markdown(
            f"<span class='badge' style='background:{color}'>{label}</span> {cnt:,}건",
            unsafe_allow_html=True,
        )


# ── 메인 헤더 ─────────────────────────────────────────────────
st.markdown("# 📰 macro-graph 기사 검색")
st.markdown("---")

# ── 검색 실행 ─────────────────────────────────────────────────
selected_sources = [k for k, v in source_checks.items() if v]

df = search_articles(
    keyword=keyword,
    date_from=date_from,
    date_to=date_to,
    sources=tuple(selected_sources),
    limit=result_limit,
)

# ── 결과 수 표시 ───────────────────────────────────────────────
total_match_info = f"검색 결과: **{len(df):,}건**"
if keyword:
    total_match_info += f" (키워드: `{keyword}`)"
st.markdown(total_match_info)

if df.empty:
    st.info("검색 결과가 없습니다. 검색 조건을 바꿔보세요.")
else:
    # ── 탭 분리: 카드뷰 / 테이블뷰 ────────────────────────────
    tab1, tab2 = st.tabs(["🗂 카드 보기", "📋 테이블 보기"])

    with tab1:
        for _, row in df.iterrows():
            src_label = SOURCE_MAP.get(row["source"], row["source"])
            color = SOURCE_COLOR.get(src_label, "#999")

            body_preview = ""
            if row["body"]:
                raw = str(row["body"]).replace("\n", " ").replace("\r", " ")
                body_preview = raw[:200] + ("..." if len(raw) > 200 else "")

            st.markdown(f"""
<div class="article-card">
  <div>
    <span class="badge" style="background:{color}">{src_label}</span>
    <span class="art-date">{row['published_at']}</span>
  </div>
  <div class="art-title">{row['title']}</div>
  {"<div class='art-body'>" + body_preview + "</div>" if body_preview else ""}
  <a class="art-link" href="{row['url']}" target="_blank">🔗 기사 원문 →</a>
</div>
""", unsafe_allow_html=True)

    with tab2:
        display_df = df[["published_at", "source", "title", "url"]].copy()
        display_df["source"] = display_df["source"].map(SOURCE_MAP).fillna(display_df["source"])
        display_df.columns = ["날짜", "출처", "제목", "URL"]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            column_config={
                "URL": st.column_config.LinkColumn("URL", display_text="바로가기"),
            },
        )

        # CSV 다운로드
        csv = display_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ CSV로 내보내기",
            data=csv,
            file_name=f"articles_{keyword or 'all'}.csv",
            mime="text/csv",
        )
