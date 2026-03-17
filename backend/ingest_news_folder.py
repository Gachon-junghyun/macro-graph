"""
news/ 폴더 안의 .txt 파일을 읽어 DB에 저장하는 스크립트.

txt 파일 형식:
─────────────────────────────
제목: 기사 제목
날짜: 2026-03-10          ← 생략 가능 (생략 시 오늘 날짜)
---
본문 내용
여러 줄 가능
─────────────────────────────

실행 방법:
    cd macro-graph/backend
    python ingest_news_folder.py

API로 실행 (서버 켜진 상태):
    curl -X POST http://localhost:8000/api/pipeline/ingest-news
"""

import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from database import init_db, get_db

NEWS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news")


# ── txt 파싱 ──────────────────────────────────────────────────
def parse_news_file(filepath: str) -> Optional[Dict]:
    """
    txt 파일 하나를 파싱해서 기사 딕셔너리 반환.
    형식이 맞지 않으면 None 반환.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return None

        # --- 구분자 기준으로 헤더/본문 분리
        if "---" in content:
            header_part, body_part = content.split("---", 1)
        else:
            # 구분자 없으면 첫 줄을 제목, 나머지를 본문으로 처리
            lines = content.splitlines()
            header_part = lines[0]
            body_part = "\n".join(lines[1:])

        # 헤더 파싱
        title = ""
        date_str = datetime.now().strftime("%Y-%m-%d")

        for line in header_part.splitlines():
            line = line.strip()
            if line.startswith("제목:"):
                title = line[len("제목:"):].strip()
            elif line.startswith("날짜:"):
                date_str = line[len("날짜:"):].strip()

        # 제목이 없으면 파일명을 제목으로 사용
        if not title:
            title = os.path.splitext(os.path.basename(filepath))[0]

        body = body_part.strip()

        if len(title) < 2:
            print(f"  [SKIP] 제목이 너무 짧음: {filepath}")
            return None

        return {
            "source":       "manual_txt",
            "title":        title,
            "body":         body[:5000],
            "published_at": date_str,
        }

    except Exception as e:
        print(f"  [ERROR] 파일 파싱 실패 ({filepath}): {e}")
        return None


# ── 중복 확인 (제목 해시 기반) ────────────────────────────────
def _make_manual_url(title: str, filename: str) -> str:
    """
    수동 입력 기사는 URL이 없으므로 제목+파일명 해시로 가짜 URL 생성.
    DB의 url UNIQUE 제약을 중복 방지에 활용.
    """
    key = f"{filename}::{title}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"manual://{h}"


def _is_duplicate(conn, url: str) -> bool:
    row = conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone()
    return row is not None


# ── 폴더 전체 처리 ────────────────────────────────────────────
def ingest_news_folder(folder: str = NEWS_FOLDER) -> Dict:
    """
    news/ 폴더 안의 모든 .txt 파일을 읽어 DB에 저장.

    중복 처리:
      1) _로 시작하는 파일(샘플 등) 건너뜀
      2) 같은 제목+파일명 조합은 해시 URL로 DB 중복 체크
    """
    if not os.path.exists(folder):
        return {"error": f"폴더 없음: {folder}"}

    txt_files = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".txt") and not f.startswith("_")   # _로 시작하는 파일 제외
    ])

    if not txt_files:
        return {"message": "처리할 .txt 파일이 없습니다.", "folder": folder}

    print("=" * 50)
    print(f"[news 폴더] {len(txt_files)}개 파일 처리 시작")

    parsed_ok, skipped_dup, skipped_parse, saved = 0, 0, 0, 0
    articles_to_save: List[Dict] = []

    with get_db() as conn:
        for filename in txt_files:
            filepath = os.path.join(folder, filename)
            article = parse_news_file(filepath)

            if article is None:
                skipped_parse += 1
                continue

            # 중복 체크
            url = _make_manual_url(article["title"], filename)
            if _is_duplicate(conn, article["title_url"] if "title_url" in article else url):
                print(f"  [SKIP-DB] {filename}")
                skipped_dup += 1
                continue

            article["url"] = url
            articles_to_save.append(article)
            parsed_ok += 1
            print(f"  [OK] {filename} → {article['title'][:50]}")

    # DB 저장
    with get_db() as conn:
        for art in articles_to_save:
            # 한 번 더 중복 확인 (혹시 같은 실행 내 중복 파일이 있을 경우)
            if _is_duplicate(conn, art["url"]):
                skipped_dup += 1
                continue
            try:
                conn.execute(
                    """INSERT INTO articles (source, title, body, url, published_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (art["source"], art["title"], art["body"],
                     art["url"], art["published_at"]),
                )
                saved += 1
            except Exception as e:
                print(f"  [ERROR] 저장 실패: {e}")

    print(f"\n결과: 파싱성공 {parsed_ok}개 / 중복건너뜀 {skipped_dup}개 "
          f"/ 파싱실패 {skipped_parse}개 / 저장 {saved}개")
    print("=" * 50)

    return {
        "total_files":    len(txt_files),
        "parsed_ok":      parsed_ok,
        "skipped_dup":    skipped_dup,
        "skipped_parse":  skipped_parse,
        "saved":          saved,
        "folder":         folder,
    }


if __name__ == "__main__":
    init_db()
    result = ingest_news_folder()
    print(result)
