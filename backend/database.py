"""SQLite database setup and helper functions."""
import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "macro_graph.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    # 마이그레이션을 먼저 실행해야 executescript의 인덱스 생성이 안전함
    _migrate(conn_or_path=DB_PATH)

    with get_db() as conn:
        conn.executescript("""
        -- 1단계: 기사 테이블
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT,
            url TEXT UNIQUE,
            published_at TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_articles_published
            ON articles(published_at);
        CREATE INDEX IF NOT EXISTS idx_articles_url
            ON articles(url);

        -- 1단계: 기사별 명사 테이블
        CREATE TABLE IF NOT EXISTS article_nouns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            noun TEXT NOT NULL,
            position TEXT NOT NULL,  -- 'title' or 'body'
            FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_article_nouns_article
            ON article_nouns(article_id);
        CREATE INDEX IF NOT EXISTS idx_article_nouns_noun
            ON article_nouns(noun);

        -- 2단계: 노드 테이블
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            noun TEXT UNIQUE NOT NULL,
            total_count INTEGER DEFAULT 0,
            last_seen TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_noun
            ON nodes(noun);

        -- 2단계: 엣지 테이블
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_a INTEGER NOT NULL,
            node_b INTEGER NOT NULL,
            score REAL DEFAULT 0,
            article_count INTEGER DEFAULT 0,
            last_seen TEXT,
            FOREIGN KEY (node_a) REFERENCES nodes(id),
            FOREIGN KEY (node_b) REFERENCES nodes(id),
            UNIQUE(node_a, node_b)
        );

        CREATE INDEX IF NOT EXISTS idx_edges_nodes
            ON edges(node_a, node_b);
        CREATE INDEX IF NOT EXISTS idx_edges_score
            ON edges(score);

        -- 4단계: 가격 데이터 테이블
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            asset_name TEXT NOT NULL,
            date TEXT NOT NULL,
            close_price REAL,
            UNIQUE(ticker, date)
        );

        -- 인과 관계 엣지 (Gemini 추출, 방향 있음)
        CREATE TABLE IF NOT EXISTS causal_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cause TEXT NOT NULL,         -- 원인 노드 명사
            effect TEXT NOT NULL,        -- 결과 노드 명사
            relation TEXT,               -- 관계 설명 (15자 이내)
            strength INTEGER DEFAULT 1,  -- 1=약 2=중 3=강
            article_id INTEGER,
            category TEXT,               -- 체인 카테고리 (예: 실물경제, 지정학)
            chain_text TEXT,             -- 전체 체인 텍스트 (예: A → B → C → D)
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (article_id) REFERENCES articles(id),
            UNIQUE(cause, effect, article_id)
        );

        CREATE INDEX IF NOT EXISTS idx_causal_cause
            ON causal_edges(cause);
        CREATE INDEX IF NOT EXISTS idx_causal_effect
            ON causal_edges(effect);
        CREATE INDEX IF NOT EXISTS idx_causal_category
            ON causal_edges(category);

        -- 인과 체인 전체 저장 (체인 단위 원본 보존)
        CREATE TABLE IF NOT EXISTS causal_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            category TEXT NOT NULL,
            chain_text TEXT NOT NULL,    -- "A → B → C → D" 형태
            confidence INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (article_id) REFERENCES articles(id),
            UNIQUE(chain_text, article_id)
        );

        CREATE INDEX IF NOT EXISTS idx_chains_category
            ON causal_chains(category);
        CREATE INDEX IF NOT EXISTS idx_chains_article
            ON causal_chains(article_id);

        -- 4단계: 언급 급증 이벤트 테이블
        CREATE TABLE IF NOT EXISTS mention_spikes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            noun TEXT NOT NULL,
            spike_date TEXT NOT NULL,
            mention_count INTEGER,
            prev_avg_count REAL,
            return_1d REAL,
            return_7d REAL,
            return_30d REAL,
            UNIQUE(noun, spike_date)
        );
        """)
    # 기존 DB 마이그레이션: causal_edges에 컬럼 추가 (없을 때만)
    _migrate(conn_or_path=DB_PATH)
    print("Database initialized at", DB_PATH)


def _migrate(conn_or_path: str):
    """기존 DB에 새 컬럼/인덱스/테이블을 안전하게 추가 (idempotent).
    init_db()의 executescript보다 반드시 먼저 실행되어야 함.
    """
    conn = sqlite3.connect(conn_or_path)
    try:
        # causal_edges 테이블이 아직 없으면 마이그레이션 불필요
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='causal_edges'"
        ).fetchone()
        if not exists:
            return

        # category 컬럼 추가 + 인덱스
        try:
            conn.execute("ALTER TABLE causal_edges ADD COLUMN category TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # 이미 존재

        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_causal_category ON causal_edges(category)"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass

        # chain_text 컬럼 추가
        try:
            conn.execute("ALTER TABLE causal_edges ADD COLUMN chain_text TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        # ── v2 마이그레이션: 정규화 레이어 테이블 ──────────────────────
        # concepts: 정규화된 표준 개념 레지스트리
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical  TEXT UNIQUE NOT NULL,
                category   TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

        # concept_aliases: raw_noun → canonical 매핑
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concept_aliases (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_noun   TEXT UNIQUE NOT NULL,
                concept_id INTEGER NOT NULL,
                method     TEXT DEFAULT 'exact',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (concept_id) REFERENCES concepts(id)
            )
        """)
        conn.commit()

        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_aliases_raw ON concept_aliases(raw_noun)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_aliases_concept ON concept_aliases(concept_id)"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass

    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
