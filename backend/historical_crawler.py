#!/usr/bin/env python3
"""
historical_crawler.py — 날짜 범위 + 검색어 기반 과거 기사 수집 크롤러
=====================================================================

현재 crawler.py 는 RSS(최신 80건) + 목록 페이지(오늘치) 만 수집합니다.
이 모듈은 **과거 데이터**를 4가지 전략으로 수집합니다:

  [S] search_page  — 각 언론사 검색 페이지 (키워드 + 날짜 파라미터 지원)
  [A] date_archive — 날짜별 아카이브 페이지 순회 (키워드 없이 전량 수집)
  [G] google_rss   — Google News RSS (키워드 + 날짜 필터, 무료 · API키 불필요)
  [N] naver_api    — 네이버 뉴스 검색 API (키워드 + 날짜 정렬, Client ID/Secret 필요)

수집된 URL은 기존 crawler.py 의 crawl_single_url() + save_articles() 를
그대로 재사용합니다 — 중복 제거, DB 저장 로직을 이중으로 구현하지 않습니다.

────────────────────────────────────────
  사용법 예시
────────────────────────────────────────
  # 최근 7일, 모든 소스
  python historical_crawler.py --days 7

  # 검색어 지정
  python historical_crawler.py --days 7 --keyword "금리"
  python historical_crawler.py --days 7 --keyword "반도체 수출"

  # 날짜 범위 직접 지정
  python historical_crawler.py --from 2026-03-09 --to 2026-03-16

  # 특정 소스만 (쉼표 구분)
  python historical_crawler.py --days 7 --sources hankyung,mk,yna

  # 소스 목록 확인
  python historical_crawler.py --list-sources

  # 드라이런 — URL 수집만, DB 저장 안 함
  python historical_crawler.py --days 7 --dry-run

  # 네이버 API 사용 (API키 필요)
  python historical_crawler.py --days 7 --keyword "반도체" \\
      --naver-id YOUR_CLIENT_ID --naver-secret YOUR_CLIENT_SECRET

  # Google News RSS (키워드 필수)
  python historical_crawler.py --days 7 --keyword "미국 관세" --sources google

  # 최대 수집 기사 수 제한
  python historical_crawler.py --days 7 --max-articles 200

────────────────────────────────────────
  소스 추가 방법
────────────────────────────────────────
  SEARCH_SOURCES 딕셔너리에 항목 하나 추가하면 됩니다.
  필수 키: strategy / name / source / search_url / date_fmt
           link_selectors / base_url / pagination / max_pages
  선택 키: requires_keyword / extra_headers / delay_sec

────────────────────────────────────────
  알려진 제한사항
────────────────────────────────────────
  - 네이버 뉴스 본문: VM 환경에서 봇 차단 → URL 수집 후 본문 크롤링 실패 가능
  - 일부 언론사 검색 결과 페이지: JS 렌더링 필요 시 목록 추출 실패
  - 셀렉터가 변경된 경우: link_selectors 수동 업데이트 필요
  - Naver API 날짜 범위 파라미터: 무료 플랜에선 미지원 → 클라이언트단 필터링
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Generator, List, Optional
from urllib.parse import urlencode, urljoin, urlparse, quote_plus

import feedparser
import requests
from bs4 import BeautifulSoup

# ── 기존 crawler.py 재사용 ─────────────────────────────────────
# 개별 기사 본문 크롤링 + DB 저장 로직을 이중으로 구현하지 않습니다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawler import HEADERS, _fetch_with_retry, crawl_single_url, save_articles
from database import get_db

# ─────────────────────────────────────────────────────────────
#  로깅
# ─────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_crawler.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("historical_crawler")


# ══════════════════════════════════════════════════════════════
#  날짜 유틸리티
# ══════════════════════════════════════════════════════════════

def _fmt(dt: datetime, fmt: str) -> str:
    """datetime을 원하는 포맷 문자열로 변환."""
    return dt.strftime(fmt)


def _date_range(from_dt: datetime, to_dt: datetime) -> Generator[datetime, None, None]:
    """from_dt ~ to_dt 사이 날짜를 하루씩 yield."""
    cur = from_dt
    while cur <= to_dt:
        yield cur
        cur += timedelta(days=1)


def _parse_date(s: str) -> datetime:
    """'YYYY-MM-DD' 또는 'YYYYMMDD' 문자열 → datetime."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"날짜 형식을 인식할 수 없습니다: {s!r}  (YYYY-MM-DD 또는 YYYYMMDD 사용)")


def _is_in_range(date_str: str, from_dt: datetime, to_dt: datetime) -> bool:
    """published_at 문자열이 날짜 범위 안에 있으면 True."""
    if not date_str:
        return True  # 날짜 불명 → 일단 포함
    try:
        dt = _parse_date(date_str[:10])
        return from_dt.date() <= dt.date() <= to_dt.date()
    except ValueError:
        return True


# ══════════════════════════════════════════════════════════════
#  소스 설정 (여기만 수정하면 됩니다)
# ══════════════════════════════════════════════════════════════
#
# 각 항목 설명:
#   strategy        : "search_page" | "date_archive" | "google_rss" | "naver_api"
#   name            : 로그 표시용 사람이 읽기 좋은 이름
#   source          : DB articles.source 에 저장될 값 (기존 crawler.py 와 동일하게)
#   search_url      : URL 템플릿. {keyword}/{from_date}/{to_date}/{page} 플레이스홀더 사용
#   date_fmt        : 날짜 포맷 (예: "%Y%m%d", "%Y-%m-%d")
#   link_selectors  : 기사 링크 추출용 CSS 셀렉터 목록 (우선순위 순)
#   base_url        : 상대 경로를 절대 경로로 변환할 때 사용
#   pagination      : True면 page={n} 파라미터로 페이지 순회
#   max_pages       : 최대 페이지 수 (실제 기사 없으면 중간에 중단)
#   requires_keyword: True면 --keyword 없이는 이 소스를 건너뜀
#   delay_sec       : 요청 간 대기 시간 (초)
#   extra_headers   : 이 소스에만 추가할 HTTP 헤더
#   no_results_text : 검색 결과 없음을 판단하는 페이지 내 텍스트 (pagination 조기 종료용)
#
SEARCH_SOURCES: Dict[str, Dict] = {

    # ── [S] search_page 소스들 ─────────────────────────────────

    "yna": {
        "strategy":        "search_page",
        "name":            "연합뉴스 검색",
        "source":          "yonhap",
        "search_url":      (
            "https://www.yna.co.kr/search/index"
            "?query={keyword}"
            "&period=custom&from={from_date}&to={to_date}"
            "&sortType=latest&page={page}"
        ),
        "date_fmt":        "%Y%m%d",
        # 검색 결과 목록에서 기사 링크 추출
        "link_selectors":  [
            "ul.list a[href*='/view/']",
            ".news-con a",
            "div.yna-news-list a[href*='/view/']",
            "a[href*='yna.co.kr/view']",
        ],
        "base_url":        "https://www.yna.co.kr",
        "pagination":      True,
        "max_pages":       10,
        "requires_keyword":False,
        "delay_sec":       0.4,
        "no_results_text": "검색 결과가 없습니다",
    },

    "hankyung": {
        "strategy":        "search_page",
        "name":            "한국경제 검색",
        "source":          "hankyung",
        "search_url":      (
            "https://search.hankyung.com/apps.frm/search.do"
            "?keyword={keyword}"
            "&section=news"
            "&startDate={from_date}&endDate={to_date}"
            "&sortby=date&page={page}"
        ),
        "date_fmt":        "%Y%m%d",
        "link_selectors":  [
            "a.news-tit",
            "h3.tit a",
            "ul.c_list_article a[href*='hankyung.com/article']",
            ".search-result a[href*='hankyung.com']",
        ],
        "base_url":        "https://www.hankyung.com",
        "pagination":      True,
        "max_pages":       10,
        "requires_keyword":True,   # 한경 검색은 키워드 없으면 의미 없음
        "delay_sec":       0.4,
        "no_results_text": "검색된 결과가 없습니다",
    },

    "mk": {
        "strategy":        "search_page",
        "name":            "매일경제 검색",
        "source":          "mk",
        "search_url":      (
            "https://search.mk.co.kr/search.php"
            "?word={keyword}"
            "&startdate={from_date}&enddate={to_date}"
            "&sort=date&page={page}"
        ),
        "date_fmt":        "%Y%m%d",
        "link_selectors":  [
            "dl.text_box a",
            "ul.search_news a[href*='mk.co.kr']",
            ".news-list a[href*='mk.co.kr/news']",
            "a[href*='mk.co.kr/news']",
        ],
        "base_url":        "https://www.mk.co.kr",
        "pagination":      True,
        "max_pages":       10,
        "requires_keyword":True,
        "delay_sec":       0.4,
        "no_results_text": "검색 결과가 없습니다",
    },

    "edaily": {
        "strategy":        "search_page",
        "name":            "이데일리 검색",
        "source":          "edaily",
        "search_url":      (
            "https://www.edaily.co.kr/search/result"
            "?keyword={keyword}"
            "&startDate={from_date}&endDate={to_date}"
            "&page={page}"
        ),
        "date_fmt":        "%Y-%m-%d",
        "link_selectors":  [
            "a[href*='edaily.co.kr/news/read']",
            "a[href*='/news/newsRead']",
            ".item-news a",
            "ul.news-list a",
        ],
        "base_url":        "https://www.edaily.co.kr",
        "pagination":      True,
        "max_pages":       8,
        "requires_keyword":True,
        "delay_sec":       0.4,
        "no_results_text": "검색 결과가 없습니다",
    },

    "mt": {
        "strategy":        "search_page",
        "name":            "머니투데이 검색",
        "source":          "mt",
        "search_url":      (
            "https://search.mt.co.kr/mtview.php"
            "?SEARCH_WORD={keyword}"
            "&SEARCH_START_DATE={from_date}&SEARCH_END_DATE={to_date}"
            "&page={page}"
        ),
        "date_fmt":        "%Y%m%d",
        "link_selectors":  [
            "a[href*='mt.co.kr/mtview']",
            "a[href*='news.mt.co.kr']",
            ".news_list a",
            "ul.list_news a",
        ],
        "base_url":        "https://news.mt.co.kr",
        "pagination":      True,
        "max_pages":       8,
        "requires_keyword":True,
        "delay_sec":       0.4,
        "no_results_text": "검색된 기사가 없습니다",
    },

    "heraldcorp": {
        "strategy":        "search_page",
        "name":            "헤럴드경제 검색",
        "source":          "heraldcorp",
        "search_url":      (
            "https://biz.heraldcorp.com/search/index.php"
            "?q={keyword}"
            "&startDate={from_date}&endDate={to_date}"
            "&page={page}"
        ),
        "date_fmt":        "%Y-%m-%d",
        "link_selectors":  [
            "a[href*='heraldcorp.com/view']",
            "a[href*='/view/']",
            ".news-list a",
            "ul.search-result a",
        ],
        "base_url":        "https://biz.heraldcorp.com",
        "pagination":      True,
        "max_pages":       8,
        "requires_keyword":True,
        "delay_sec":       0.4,
        "no_results_text": "검색 결과가 없습니다",
    },

    "einfomax": {
        "strategy":        "search_page",
        "name":            "이인포맥스 검색",
        "source":          "einfomax",
        "search_url":      (
            "https://news.einfomax.co.kr/news/search"
            "?keyword={keyword}"
            "&startDate={from_date}&endDate={to_date}"
            "&page={page}"
        ),
        "date_fmt":        "%Y-%m-%d",
        "link_selectors":  [
            "a[href*='/news/articleView']",
            ".news-list a",
            "ul.article_list a",
        ],
        "base_url":        "https://news.einfomax.co.kr",
        "pagination":      True,
        "max_pages":       5,
        "requires_keyword":False,
        "delay_sec":       0.5,
        "no_results_text": "검색 결과가 없습니다",
    },

    "chosun": {
        "strategy":        "search_page",
        "name":            "조선비즈 검색",
        "source":          "chosun",
        "search_url":      (
            "https://search.chosun.com/search.php"
            "?query={keyword}"
            "&sort=date"
            "&startDate={from_date}&endDate={to_date}"
            "&page={page}"
        ),
        "date_fmt":        "%Y-%m-%d",
        "link_selectors":  [
            "a[href*='biz.chosun.com']",
            ".search-result a",
            "ul.result-list a",
        ],
        "base_url":        "https://biz.chosun.com",
        "pagination":      True,
        "max_pages":       8,
        "requires_keyword":True,
        "delay_sec":       0.4,
        "no_results_text": "검색 결과가 없습니다",
    },

    # ── [A] date_archive 소스들 (키워드 없이 날짜별 전량 수집) ──

    "yna_archive": {
        "strategy":        "date_archive",
        "name":            "연합뉴스 경제 아카이브",
        "source":          "yonhap",
        "archive_url":     "https://www.yna.co.kr/news/economy?date={date}",
        "date_fmt":        "%Y%m%d",
        "link_selectors":  [
            ".news-con a[href*='/view/']",
            "ul.list a[href*='/view/']",
            "a[href*='yna.co.kr/view']",
        ],
        "base_url":        "https://www.yna.co.kr",
        "delay_sec":       0.4,
        "no_results_text": "등록된 기사가 없습니다",
    },

    "mk_archive": {
        "strategy":        "date_archive",
        "name":            "매일경제 경제 아카이브",
        "source":          "mk",
        # mk는 날짜 파라미터가 /news/economy/?date=YYYYMMDD 패턴
        "archive_url":     "https://www.mk.co.kr/news/economy/?date={date}",
        "date_fmt":        "%Y%m%d",
        "link_selectors":  [
            "a[href*='/news/economy/']",
            "a[href*='/news/stock/']",
            "h3.news_ttl a",
            "a.news_item",
        ],
        "base_url":        "https://www.mk.co.kr",
        "delay_sec":       0.4,
        "no_results_text": "해당 날짜에 기사가 없습니다",
    },

    "hankyung_archive": {
        "strategy":        "date_archive",
        "name":            "한국경제 경제 아카이브",
        "source":          "hankyung",
        # 한경 날짜별 아카이브 패턴
        "archive_url":     "https://www.hankyung.com/economy?date={date}",
        "date_fmt":        "%Y%m%d",
        "link_selectors":  [
            "a[href*='hankyung.com/article']",
            "h3.tit a",
            ".news-tit a",
        ],
        "base_url":        "https://www.hankyung.com",
        "delay_sec":       0.4,
        "no_results_text": "등록된 기사가 없습니다",
    },

    # ── [G] Google News RSS (키워드 필수) ──────────────────────
    # API 키 불필요. 한국어 뉴스 전체를 키워드 + 날짜 필터로 수집.
    # Google News 링크를 실제 언론사 URL로 역추적합니다.

    "google": {
        "strategy":        "google_rss",
        "name":            "Google 뉴스 RSS",
        "source":          "google_news",   # 역추적 후 도메인별 source 덮어씀
        "rss_url":         (
            "https://news.google.com/rss/search"
            "?q={keyword}+after:{from_date}+before:{to_date}"
            "&hl=ko&gl=KR&ceid=KR:ko"
        ),
        "date_fmt":        "%Y-%m-%d",
        "requires_keyword":True,
        "delay_sec":       0.5,
    },

    # ── [N] Naver News API (API키 필요) ────────────────────────
    # 클라이언트 ID/Secret 필요 (무료 발급: developers.naver.com)
    # 주의: 무료 플랜은 날짜 범위 파라미터 미지원 → 클라이언트 필터링

    "naver_api": {
        "strategy":         "naver_api",
        "name":             "네이버 뉴스 API",
        "source":           "naver",
        "api_url":          "https://openapi.naver.com/v1/search/news.json",
        "max_items":        1000,   # API 총 수집 한도 (page당 100 × 10회)
        "requires_keyword": True,
        "delay_sec":        0.2,
    },
}


# ══════════════════════════════════════════════════════════════
#  URL 수집 — 공통 유틸리티
# ══════════════════════════════════════════════════════════════

def _extract_links(soup: BeautifulSoup, selectors: List[str], base_url: str) -> List[str]:
    """
    CSS 셀렉터 목록을 순서대로 시도해 기사 링크를 추출.
    상대 경로는 base_url 기준 절대 경로로 변환.
    """
    links: List[str] = []
    seen: set = set()

    for sel in selectors:
        try:
            for a in soup.select(sel):
                href = a.get("href", "").strip()
                if not href:
                    continue
                if not href.startswith("http"):
                    href = urljoin(base_url, href)
                # 쿼리스트링 제거 후 중복 체크
                clean = href.split("?")[0].rstrip("/")
                if clean in seen:
                    continue
                seen.add(clean)
                links.append(href)
        except Exception as e:
            log.debug(f"셀렉터 오류 [{sel}]: {e}")
            continue

    return links


def _is_db_duplicate(url: str) -> bool:
    """DB에 이미 있는 URL이면 True."""
    try:
        with get_db() as conn:
            row = conn.execute("SELECT 1 FROM articles WHERE url=?", (url,)).fetchone()
            return row is not None
    except Exception:
        return False


def _filter_urls(urls: List[str]) -> List[str]:
    """중복 URL + DB 기존 URL 제거 후 반환."""
    seen: set = set()
    result: List[str] = []
    for url in urls:
        key = url.split("?")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        if _is_db_duplicate(url):
            log.debug(f"[SKIP-DB] {url}")
            continue
        result.append(url)
    return result


# ══════════════════════════════════════════════════════════════
#  전략 1: search_page — 언론사 검색 페이지 순회
# ══════════════════════════════════════════════════════════════

def _collect_search_page(
    cfg: Dict,
    keyword: str,
    from_dt: datetime,
    to_dt: datetime,
) -> List[str]:
    """
    언론사 검색 페이지에서 날짜 범위 내 기사 URL 수집.
    페이지네이션 지원: 결과 없으면 조기 종료.
    """
    urls: List[str] = []
    name       = cfg["name"]
    date_fmt   = cfg["date_fmt"]
    base_url   = cfg["base_url"]
    selectors  = cfg["link_selectors"]
    max_pages  = cfg.get("max_pages", 5)
    delay      = cfg.get("delay_sec", 0.4)
    no_result  = cfg.get("no_results_text", "")

    from_str = _fmt(from_dt, date_fmt)
    to_str   = _fmt(to_dt, date_fmt)
    kw_enc   = quote_plus(keyword) if keyword else ""

    for page in range(1, max_pages + 1):
        url = cfg["search_url"].format(
            keyword=kw_enc,
            from_date=from_str,
            to_date=to_str,
            page=page,
        )
        log.info(f"  [{name}] 페이지 {page}: {url}")

        resp = _fetch_with_retry(url, max_retries=2)
        if not resp:
            log.warning(f"  [{name}] 응답 없음 — 중단")
            break

        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        # 검색 결과 없음 판단
        if no_result and no_result in soup.get_text():
            log.info(f"  [{name}] 검색 결과 없음 (p.{page}) — 종료")
            break

        page_links = _extract_links(soup, selectors, base_url)
        if not page_links:
            log.info(f"  [{name}] 링크 없음 (p.{page}) — 종료")
            break

        log.info(f"  [{name}] {len(page_links)}개 링크 발견 (p.{page})")
        urls.extend(page_links)
        time.sleep(delay)

    return urls


# ══════════════════════════════════════════════════════════════
#  전략 2: date_archive — 날짜별 아카이브 페이지 순회
# ══════════════════════════════════════════════════════════════

def _collect_date_archive(
    cfg: Dict,
    from_dt: datetime,
    to_dt: datetime,
) -> List[str]:
    """
    날짜별 아카이브 페이지를 하루씩 순회해 기사 URL 수집.
    """
    urls: List[str] = []
    name      = cfg["name"]
    date_fmt  = cfg["date_fmt"]
    base_url  = cfg["base_url"]
    selectors = cfg["link_selectors"]
    delay     = cfg.get("delay_sec", 0.4)
    no_result = cfg.get("no_results_text", "")

    total_days = (to_dt - from_dt).days + 1
    log.info(f"  [{name}] {total_days}일 아카이브 순회 시작")

    for day in _date_range(from_dt, to_dt):
        date_str = _fmt(day, date_fmt)
        url = cfg["archive_url"].format(date=date_str)
        log.info(f"  [{name}] {date_str}: {url}")

        resp = _fetch_with_retry(url, max_retries=2)
        if not resp:
            log.warning(f"  [{name}] {date_str} 응답 없음 — 건너뜀")
            time.sleep(delay)
            continue

        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        if no_result and no_result in soup.get_text():
            log.info(f"  [{name}] {date_str} 기사 없음")
            time.sleep(delay)
            continue

        day_links = _extract_links(soup, selectors, base_url)
        log.info(f"  [{name}] {date_str}: {len(day_links)}개 링크")
        urls.extend(day_links)
        time.sleep(delay)

    return urls


# ══════════════════════════════════════════════════════════════
#  전략 3: google_rss — Google News RSS
# ══════════════════════════════════════════════════════════════

def _resolve_google_redirect(google_url: str, timeout: int = 8) -> str:
    """
    Google News 리다이렉트 URL → 실제 언론사 URL 추적.
    실패 시 원본 URL 그대로 반환.
    """
    try:
        resp = requests.get(
            google_url,
            headers=HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        final = resp.url
        # Google 도메인에 머물면 HTML에서 정규 URL 찾기 시도
        if "google.com" in final:
            m = re.search(r'url=([^&"\']+)', resp.text)
            if m:
                from urllib.parse import unquote
                return unquote(m.group(1))
        return final
    except Exception:
        return google_url


def _collect_google_rss(
    cfg: Dict,
    keyword: str,
    from_dt: datetime,
    to_dt: datetime,
) -> List[str]:
    """
    Google News RSS에서 기사 URL 수집 후 원본 언론사 URL로 역추적.
    Google News는 after:/before: 연산자로 날짜 필터링 지원.
    """
    urls: List[str] = []
    date_fmt = cfg["date_fmt"]
    delay    = cfg.get("delay_sec", 0.5)

    rss_url = cfg["rss_url"].format(
        keyword=quote_plus(keyword),
        from_date=_fmt(from_dt, date_fmt),
        to_date=_fmt(to_dt + timedelta(days=1), date_fmt),   # before는 당일 미포함 → +1일
    )
    log.info(f"  [Google RSS] {rss_url}")

    feed = feedparser.parse(rss_url)
    if not feed.entries:
        log.warning("  [Google RSS] 항목 없음 (봇 차단 또는 결과 없음)")
        return []

    log.info(f"  [Google RSS] {len(feed.entries)}개 항목 수집")

    for entry in feed.entries:
        raw_url = entry.get("link", "")
        if not raw_url:
            continue

        # Google 리다이렉트 URL 역추적
        if "news.google.com" in raw_url:
            real_url = _resolve_google_redirect(raw_url)
            log.debug(f"  [Google] 역추적: {real_url}")
        else:
            real_url = raw_url

        # 날짜 필터링 (feedparser가 파싱한 날짜 사용)
        pub = entry.get("published_parsed")
        if pub:
            try:
                pub_dt = datetime(*pub[:6])
                if not (from_dt.date() <= pub_dt.date() <= to_dt.date()):
                    continue
            except Exception:
                pass

        urls.append(real_url)
        time.sleep(delay * 0.3)   # RSS이므로 빠르게

    return urls


# ══════════════════════════════════════════════════════════════
#  전략 4: naver_api — 네이버 뉴스 검색 API
# ══════════════════════════════════════════════════════════════

def _collect_naver_api(
    cfg: Dict,
    keyword: str,
    from_dt: datetime,
    to_dt: datetime,
    client_id: str,
    client_secret: str,
) -> List[str]:
    """
    네이버 뉴스 검색 API 로 기사 URL 수집.
    - 무료 플랜: 날짜 범위 파라미터 미지원 → 클라이언트단 필터링
    - originallink (원본 언론사 URL) 우선, 없으면 link (네이버 URL)
    - 네이버 URL은 본문 크롤링 실패 가능성 높음 (봇 차단)
    """
    urls: List[str] = []
    api_url   = cfg["api_url"]
    max_items = cfg.get("max_items", 1000)
    delay     = cfg.get("delay_sec", 0.2)

    headers = {
        "X-Naver-Client-Id":     client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    display = 100   # 1회 최대 100건
    collected = 0

    for start in range(1, max_items + 1, display):
        params = {
            "query":   keyword,
            "display": display,
            "start":   start,
            "sort":    "date",   # 최신순 정렬
        }
        req_url = api_url + "?" + urlencode(params)
        log.info(f"  [Naver API] start={start}: {keyword}")

        try:
            resp = requests.get(req_url, headers=headers, timeout=10)
            if resp.status_code == 401:
                log.error("  [Naver API] 인증 실패 — Client ID/Secret 확인")
                break
            if resp.status_code != 200:
                log.warning(f"  [Naver API] HTTP {resp.status_code}")
                break

            data = resp.json()
        except Exception as e:
            log.error(f"  [Naver API] 오류: {e}")
            break

        items = data.get("items", [])
        if not items:
            log.info("  [Naver API] 더 이상 결과 없음 — 종료")
            break

        for item in items:
            # 날짜 필터링 (클라이언트단)
            pub_raw = item.get("pubDate", "")
            if pub_raw:
                try:
                    # RFC 2822 형식: "Sun, 16 Mar 2026 09:00:00 +0900"
                    import email.utils
                    pub_dt = datetime(*email.utils.parsedate(pub_raw)[:6])
                    if not (from_dt.date() <= pub_dt.date() <= to_dt.date()):
                        continue
                except Exception:
                    pass

            # 원본 언론사 URL 우선
            url = item.get("originallink") or item.get("link", "")
            if url:
                urls.append(url)
                collected += 1

        log.info(f"  [Naver API] {len(items)}건 수집 (누계 {collected})")
        time.sleep(delay)

        # API 총 결과 수보다 많이 요청하지 않도록
        total = data.get("total", 0)
        if start + display > min(total, max_items):
            break

    return urls


# ══════════════════════════════════════════════════════════════
#  URL → 기사 본문 수집 (기존 crawler.py 재사용)
# ══════════════════════════════════════════════════════════════

def _crawl_urls(
    urls: List[str],
    dry_run: bool = False,
    delay_sec: float = 0.4,
    max_articles: Optional[int] = None,
) -> Dict:
    """
    URL 목록에서 기사 본문 크롤링 후 DB 저장.
    기존 crawler.py의 crawl_single_url() + save_articles() 재사용.

    Returns:
        {"crawled": n, "saved": n, "failed": n, "skipped_db": n}
    """
    if max_articles:
        urls = urls[:max_articles]

    crawled, failed, skipped = 0, 0, 0
    articles = []

    for i, url in enumerate(urls, 1):
        log.info(f"  [{i}/{len(urls)}] {url[:80]}")

        if dry_run:
            log.info("  [DRY-RUN] 크롤링 건너뜀")
            crawled += 1
            continue

        art = crawl_single_url(url)
        if art:
            articles.append(art)
            crawled += 1
        else:
            failed += 1
        time.sleep(delay_sec)

    saved = 0 if dry_run else save_articles(articles)
    return {
        "crawled":    crawled,
        "saved":      saved,
        "failed":     failed,
        "skipped_db": skipped,
    }


# ══════════════════════════════════════════════════════════════
#  메인 오케스트레이터
# ══════════════════════════════════════════════════════════════

def run_historical(
    from_dt: datetime,
    to_dt: datetime,
    keyword: str = "",
    sources: Optional[List[str]] = None,
    dry_run: bool = False,
    naver_client_id: str = "",
    naver_client_secret: str = "",
    max_articles: Optional[int] = None,
    article_delay: float = 0.4,
) -> Dict:
    """
    과거 기사 수집 실행.

    Args:
        from_dt:            수집 시작 날짜
        to_dt:              수집 종료 날짜 (포함)
        keyword:            검색어 (비워두면 search_page/google/naver 소스 건너뜀)
        sources:            수집할 소스 키 목록 (None이면 전체)
        dry_run:            True면 URL 수집만, DB 저장 안 함
        naver_client_id:    네이버 API Client ID
        naver_client_secret:네이버 API Client Secret
        max_articles:       소스별 최대 기사 수 (None이면 무제한)
        article_delay:      기사 간 크롤링 대기 시간 (초)

    Returns:
        소스별 + 전체 요약 딕셔너리
    """
    log.info("=" * 70)
    log.info(f"과거 기사 수집 시작")
    log.info(f"  기간: {from_dt.strftime('%Y-%m-%d')} ~ {to_dt.strftime('%Y-%m-%d')}")
    log.info(f"  키워드: {keyword!r} (빈 문자열이면 키워드 불필요 소스만 실행)")
    log.info(f"  드라이런: {dry_run}")

    target_sources = sources if sources else list(SEARCH_SOURCES.keys())
    summary = {
        "period":  f"{from_dt.strftime('%Y-%m-%d')} ~ {to_dt.strftime('%Y-%m-%d')}",
        "keyword": keyword,
        "dry_run": dry_run,
        "sources": {},
        "total": {"url_collected": 0, "crawled": 0, "saved": 0, "failed": 0},
    }

    for src_key in target_sources:
        cfg = SEARCH_SOURCES.get(src_key)
        if cfg is None:
            log.warning(f"알 수 없는 소스: {src_key!r} — 건너뜀")
            continue

        strategy = cfg["strategy"]
        name     = cfg["name"]

        # 키워드 필요 여부 체크
        if cfg.get("requires_keyword") and not keyword:
            log.info(f"[{name}] 키워드 없음 — 건너뜀 (--keyword 지정 필요)")
            continue

        # Naver API 키 체크
        if strategy == "naver_api" and (not naver_client_id or not naver_client_secret):
            log.info(f"[{name}] API 키 없음 — 건너뜀 (--naver-id / --naver-secret 지정 필요)")
            continue

        log.info(f"\n{'─' * 50}")
        log.info(f"[소스] {name} ({strategy})")

        # ── URL 수집 (전략별 분기) ────────────────────────────
        try:
            if strategy == "search_page":
                raw_urls = _collect_search_page(cfg, keyword or "", from_dt, to_dt)

            elif strategy == "date_archive":
                raw_urls = _collect_date_archive(cfg, from_dt, to_dt)

            elif strategy == "google_rss":
                if not keyword:
                    log.info(f"[{name}] 키워드 없음 — 건너뜀")
                    continue
                raw_urls = _collect_google_rss(cfg, keyword, from_dt, to_dt)

            elif strategy == "naver_api":
                raw_urls = _collect_naver_api(
                    cfg, keyword, from_dt, to_dt,
                    naver_client_id, naver_client_secret,
                )

            else:
                log.warning(f"[{name}] 알 수 없는 전략: {strategy}")
                continue

        except Exception as e:
            log.error(f"[{name}] URL 수집 중 오류: {e}", exc_info=True)
            continue

        log.info(f"[{name}] 원본 URL {len(raw_urls)}개 수집")

        # ── 중복 제거 ──────────────────────────────────────────
        unique_urls = _filter_urls(raw_urls)
        skipped_dup = len(raw_urls) - len(unique_urls)
        log.info(f"[{name}] 중복 제거 후 {len(unique_urls)}개 ({skipped_dup}개 중복)")

        if not unique_urls:
            summary["sources"][src_key] = {
                "url_collected": 0, "crawled": 0, "saved": 0, "failed": 0,
            }
            continue

        # ── 본문 크롤링 + DB 저장 ─────────────────────────────
        result = _crawl_urls(
            unique_urls,
            dry_run=dry_run,
            delay_sec=article_delay,
            max_articles=max_articles,
        )
        result["url_collected"] = len(unique_urls)

        log.info(
            f"[{name}] 완료 — "
            f"수집 {result['crawled']}건 / 저장 {result['saved']}건 / 실패 {result['failed']}건"
        )

        summary["sources"][src_key] = result
        summary["total"]["url_collected"] += result["url_collected"]
        summary["total"]["crawled"]       += result["crawled"]
        summary["total"]["saved"]         += result["saved"]
        summary["total"]["failed"]        += result["failed"]

    log.info(f"\n{'═' * 70}")
    log.info(f"전체 완료: {summary['total']}")
    log.info(f"{'═' * 70}\n")
    return summary


# ══════════════════════════════════════════════════════════════
#  CLI 진입점
# ══════════════════════════════════════════════════════════════

def _list_sources() -> None:
    """사용 가능한 소스 목록 출력."""
    print("\n사용 가능한 소스 목록:")
    print(f"{'키':20} {'전략':15} {'이름':25} {'키워드 필수'}")
    print("-" * 75)
    for key, cfg in SEARCH_SOURCES.items():
        req = "필수" if cfg.get("requires_keyword") else "선택"
        print(f"{key:20} {cfg['strategy']:15} {cfg['name']:25} {req}")
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="과거 뉴스 기사 수집 크롤러 (날짜 범위 + 검색어)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 날짜 범위 (상호 배타적 그룹)
    dt_group = p.add_mutually_exclusive_group()
    dt_group.add_argument(
        "--days", type=int, metavar="N",
        help="최근 N일 수집 (예: --days 7)",
    )
    dt_group.add_argument(
        "--from", dest="from_date", metavar="YYYY-MM-DD",
        help="수집 시작 날짜 (--to 와 함께 사용)",
    )

    p.add_argument(
        "--to", dest="to_date", metavar="YYYY-MM-DD",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="수집 종료 날짜 (기본: 오늘)",
    )

    # 검색어
    p.add_argument(
        "--keyword", "-k", default="",
        metavar="KEYWORD",
        help="검색어 (search_page / google / naver_api 소스에 적용)",
    )

    # 소스 선택
    p.add_argument(
        "--sources", default="",
        metavar="SRC1,SRC2,...",
        help=f"수집할 소스 키 (쉼표 구분). 기본: 전체. --list-sources 로 목록 확인",
    )

    # 네이버 API 키
    p.add_argument("--naver-id",     default="", metavar="CLIENT_ID",     help="네이버 API Client ID")
    p.add_argument("--naver-secret", default="", metavar="CLIENT_SECRET",  help="네이버 API Client Secret")

    # 기타 옵션
    p.add_argument(
        "--dry-run", action="store_true",
        help="URL 수집만 수행, DB 저장 안 함",
    )
    p.add_argument(
        "--max-articles", type=int, default=None, metavar="N",
        help="소스별 최대 기사 크롤링 수",
    )
    p.add_argument(
        "--delay", type=float, default=0.4, metavar="SEC",
        help="기사 간 크롤링 대기 시간 (초, 기본: 0.4)",
    )
    p.add_argument(
        "--list-sources", action="store_true",
        help="사용 가능한 소스 목록 출력 후 종료",
    )
    p.add_argument(
        "--output-json", metavar="PATH",
        help="결과 요약을 JSON 파일로 저장",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.list_sources:
        _list_sources()
        return

    # ── 날짜 범위 계산 ─────────────────────────────────────────
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if args.days:
        from_dt = today - timedelta(days=args.days - 1)
        to_dt   = today
    elif args.from_date:
        from_dt = _parse_date(args.from_date)
        to_dt   = _parse_date(args.to_date)
    else:
        parser.error("--days 또는 --from 중 하나를 지정하세요.")
        return

    if from_dt > to_dt:
        parser.error(f"시작 날짜({from_dt.date()})가 종료 날짜({to_dt.date()})보다 늦습니다.")

    # ── 소스 파싱 ──────────────────────────────────────────────
    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or None

    # ── 실행 ───────────────────────────────────────────────────
    from database import init_db
    init_db()

    result = run_historical(
        from_dt=from_dt,
        to_dt=to_dt,
        keyword=args.keyword,
        sources=sources,
        dry_run=args.dry_run,
        naver_client_id=args.naver_id,
        naver_client_secret=args.naver_secret,
        max_articles=args.max_articles,
        article_delay=args.delay,
    )

    # ── 결과 출력 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("수집 결과 요약")
    print("=" * 60)
    print(f"  기간    : {result['period']}")
    print(f"  키워드  : {result['keyword'] or '(없음)'}")
    print(f"  드라이런: {result['dry_run']}")
    print()
    print(f"  {'소스':20} {'URL수':>6} {'크롤링':>6} {'저장':>6} {'실패':>6}")
    print(f"  {'-'*20} {'------':>6} {'------':>6} {'------':>6} {'------':>6}")
    for src_key, r in result["sources"].items():
        name = SEARCH_SOURCES[src_key]["name"]
        print(
            f"  {name[:20]:20}"
            f" {r.get('url_collected',0):>6}"
            f" {r.get('crawled',0):>6}"
            f" {r.get('saved',0):>6}"
            f" {r.get('failed',0):>6}"
        )
    t = result["total"]
    print(f"  {'합계':20} {t['url_collected']:>6} {t['crawled']:>6} {t['saved']:>6} {t['failed']:>6}")
    print("=" * 60)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n결과 JSON 저장: {args.output_json}")


if __name__ == "__main__":
    main()
