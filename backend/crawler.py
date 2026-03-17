"""
뉴스 크롤러 — RSS + BeautifulSoup 기반 금융/경제 기사 수집.

지원 사이트 (17개):
  한국: 연합뉴스, 한국경제, 매일경제, 이데일리, 머니투데이, 헤럴드경제,
        조선비즈, 이인포맥스, 더일렉, 아이로봇뉴스, 지크서, 뉴스프라임,
        중앙일보, 동아일보, 네이버뉴스(제한적)
  글로벌: Investing.com(한국어), Investing.com(영어)

[Cloudflare 보호 사이트]
  investing.com 계열은 cloudscraper 라이브러리를 우선 사용하고,
  미설치 시 일반 requests 로 폴백합니다.
  pip install cloudscraper
"""

import re
import hashlib
import time
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import feedparser
import requests
from bs4 import BeautifulSoup

# ── cloudscraper (선택 의존성 — Cloudflare 보호 사이트용) ──────
try:
    import cloudscraper
    _cloudscraper_session = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    _cloudscraper_session = None
    CLOUDSCRAPER_AVAILABLE = False

from database import get_db

# ── 로깅 설정 ─────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crawler.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),          # 터미널에도 동시 출력
    ],
)
log = logging.getLogger("crawler")

# urls.txt 기본 경로
URL_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urls.txt")

# ── 공통 헤더 ─────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ── 사이트별 셀렉터 (title / body / source) ──────────────────
#
# 한국 주요 뉴스 사이트의 HTML 구조:
#   - 연합뉴스(yna): article.story-news / h1.tit
#   - 한경(hankyung): div.article-body / h1.headline
#   - 매경(mk): div.news_cnt_detail_wrap / h2.news_detail_headline
#   - 이데일리(edaily): div.news_body / h1.news_tit
#   - 머니투데이(mt): div#textBody / h1.tit_view
#   - 헤럴드경제(heraldcorp): div.article-body / h1.news-tit
#   - 조선비즈(chosun): section.article-body / h1
#   - 이인포맥스(einfomax): AtozBoard → div#article-view-content-div / h3.heading
#   - 더일렉(thelec): AtozBoard → div#article-view-content-div / h3.heading
#   - 아이로봇뉴스(irobotnews): AtozBoard → div#article-view-content-div / h3.heading
#   - 지크서(ziksir): AtozBoard → div#article-view-content-div / h3.heading
#   - 뉴스프라임(newsprime): AtozBoard → div#article-view-content-div / h3.heading
#   - 중앙(joongang): div#article_body / h1.headline
#   - 동아(donga): div.article_txt / h1.title
#   - 네이버뉴스: div#dic_area / h2.media_end_head_headline (봇차단 많음)
#
DOMAIN_SELECTORS: Dict[str, Dict] = {
    # ── 대형 언론 ──────────────────────────────────────────────
    "yna.co.kr": {
        "source": "yonhap",
        "title": [
            "h1.tit", "h2.tit", "div.article h1",
            "span.tit", "h1",
        ],
        "body": [
            "article.story-news", "div.article-story",
            "div.comp_article_content", "div.article",
        ],
    },
    "hankyung.com": {
        "source": "hankyung",
        "title": [
            "h1.headline", ".article-header h1", "h1.article-tit",
            "h1",
        ],
        "body": [
            "div.article-body", "div#articletxt", "div.article_body",
            "div.text",
        ],
    },
    "mk.co.kr": {
        "source": "mk",
        "title": [
            "h2.news_detail_headline", "h1.top_title", "h2.news_ttl",
            "h1",
        ],
        "body": [
            "div.news_cnt_detail_wrap", "div#article_body", "div.art_txt",
            "div.news_body",
        ],
    },
    "edaily.co.kr": {
        "source": "edaily",
        "title": [
            "h1.news_tit", "h1.title_article", "h1",
        ],
        "body": [
            "div.news_body", "div#newsBody", "div.newsBody",
            "div.article_body",
        ],
    },
    "mt.co.kr": {
        "source": "mt",
        "title": [
            "h1.tit_view", "div.article_view h1", "h1.article_tit",
            "h1",
        ],
        "body": [
            "div#textBody", "div.view_text", "div.article_content",
            "section.article_cont",
        ],
    },
    "heraldcorp.com": {
        "source": "heraldcorp",
        "title": [
            "h1.news-tit", "div.article-view h1", "h1",
        ],
        "body": [
            "div.article-body", "div#articleText", "div.view_con",
        ],
    },
    "chosun.com": {
        "source": "chosun",
        "title": [
            "h1", "h2.article-header__title", "div.article-header h1",
        ],
        "body": [
            "section.article-body", "div.article-body",
            "div[class*='article_body']",
        ],
    },
    "joongang.co.kr": {
        "source": "joongang",
        "title": [
            "h1.headline", "h1.article_title", "h1",
        ],
        "body": [
            "div#article_body", "div.article_body",
            "div[class*='article-body']",
        ],
    },
    "donga.com": {
        "source": "donga",
        "title": [
            "h1.title", "h1.article-tit", "h1",
        ],
        "body": [
            "div.article_txt", "div#content", "div.news_view_box",
        ],
    },
    # ── AtozBoard CMS 계열 (동일한 구조) ─────────────────────
    # einfomax, thelec, irobotnews, ziksir, newsprime 등
    "einfomax.co.kr": {
        "source": "einfomax",
        "title": [
            "h3.heading", "div.article-view-head h1",
            "div.view-con h3", "h1",
        ],
        "body": [
            "div#article-view-content-div", "div.article-view-content",
            "div#articleBody", "div.view-con",
        ],
    },
    "thelec.kr": {
        "source": "thelec",
        "title": [
            "h3.heading", "div.article-view-head h1", "h1",
        ],
        "body": [
            "div#article-view-content-div", "div.article-view-content",
        ],
    },
    "irobotnews.com": {
        "source": "irobotnews",
        "title": [
            "h3.heading", "div.article-view-head h1", "h1",
        ],
        "body": [
            "div#article-view-content-div", "div.article-view-content",
        ],
    },
    "ziksir.com": {
        "source": "ziksir",
        "title": [
            "h3.heading", "h1.news-title", "h1",
        ],
        "body": [
            "div#article-view-content-div", "div.article-view-content",
            "div.article-body",
        ],
    },
    "newsprime.co.kr": {
        "source": "newsprime",
        "title": [
            "h3.heading", "h2.article-title", "h1",
        ],
        "body": [
            "div#article-view-content-div", "div.article-view-content",
        ],
    },
    # ── 네이버 뉴스 ────────────────────────────────────────────
    "n.news.naver.com": {
        "source": "naver",
        "title": [
            "h2.media_end_head_headline",
            "#title_area span",
            "div.media_end_head_title h2",
            "h2[class*='head_headline']",
        ],
        "body": [
            "div#dic_area",
            "article#dic_area",
            "div.newsct_article",
            "div[class*='article_body']",
        ],
        # 네이버 전용 추가 헤더 (Referer 필요)
        "extra_headers": {
            "Referer": "https://news.naver.com/",
            "sec-fetch-site": "same-origin",
        },
    },

    # ── Investing.com (Cloudflare 보호 → cloudscraper 사용) ────
    # 한국어판: kr.investing.com  /  영어판: www.investing.com
    # RSS는 일반 feedparser 로도 수집 가능하나,
    # 개별 기사 본문 요청은 _fetch_cloudflare() 를 통해 처리됨.
    "kr.investing.com": {
        "source":    "investing_kr",
        "title":     [
            "h1.articleHeader",
            "h1[class*='articleHeader']",
            "div.articleHeader h1",
            "h1",
        ],
        "body":      [
            "div.articlePage",
            "div[class*='articlePage']",
            "div.WYSIWYG",
            "div[class*='article-content']",
            "section[class*='article']",
        ],
        "cloudflare": True,          # _fetch_cloudflare() 로 요청
        "extra_headers": {
            "Referer":  "https://kr.investing.com/news/",
            "Origin":   "https://kr.investing.com",
        },
    },
    "investing.com": {
        "source":    "investing_en",
        "title":     [
            "h1.articleHeader",
            "h1[class*='articleHeader']",
            "div.articleHeader h1",
            "h1",
        ],
        "body":      [
            "div.articlePage",
            "div[class*='articlePage']",
            "div.WYSIWYG",
            "div[class*='article-content']",
            "section[class*='article']",
        ],
        "cloudflare": True,
        "extra_headers": {
            "Referer": "https://www.investing.com/news/",
            "Origin":  "https://www.investing.com",
        },
    },
}

# ── 범용 폴백 셀렉터 (알 수 없는 사이트) ─────────────────────
FALLBACK_TITLE_SELS = [
    "h1", "h2",
    "div[class*='title'] h1", "div[class*='headline'] h1",
    "meta[property='og:title']",   # Open Graph
]
FALLBACK_BODY_SELS = [
    "article",
    "div#article-view-content-div",   # AtozBoard 공통
    "div[id*='article']", "div[id*='content']",
    "div[class*='article_body']", "div[class*='article-body']",
    "div[class*='article_content']", "div[class*='news_body']",
    "div[itemprop='articleBody']",    # Schema.org
]

# ── RSS 피드 (대용량용 — 12개 언론사, 24개 피드) ──────────────
RSS_FEEDS = [
    # 연합뉴스
    {"name": "연합 경제",    "url": "https://www.yna.co.kr/rss/economy.xml",       "source": "yonhap"},
    {"name": "연합 국제",    "url": "https://www.yna.co.kr/rss/international.xml", "source": "yonhap"},
    {"name": "연합 산업",    "url": "https://www.yna.co.kr/rss/industry.xml",      "source": "yonhap"},
    {"name": "연합 정치",    "url": "https://www.yna.co.kr/rss/politics.xml",      "source": "yonhap"},
    # 한국경제
    {"name": "한경 경제",    "url": "https://www.hankyung.com/feed/economy",       "source": "hankyung"},
    {"name": "한경 증권",    "url": "https://www.hankyung.com/feed/finance",       "source": "hankyung"},
    {"name": "한경 국제",    "url": "https://www.hankyung.com/feed/international", "source": "hankyung"},
    # 매일경제
    {"name": "매경 경제",    "url": "https://www.mk.co.kr/rss/30100041/",          "source": "mk"},
    {"name": "매경 증권",    "url": "https://www.mk.co.kr/rss/30200030/",          "source": "mk"},
    {"name": "매경 국제",    "url": "https://www.mk.co.kr/rss/30300018/",          "source": "mk"},
    # 이데일리
    {"name": "이데일리 경제","url": "https://www.edaily.co.kr/rss/economy.xml",    "source": "edaily"},
    {"name": "이데일리 증권","url": "https://www.edaily.co.kr/rss/stock.xml",      "source": "edaily"},
    # 머니투데이
    {"name": "머니투데이",   "url": "https://www.mt.co.kr/rss/mt_stock.xml",       "source": "mt"},
    # 헤럴드경제
    {"name": "헤럴드경제",   "url": "https://biz.heraldcorp.com/rss/finance.xml",  "source": "heraldcorp"},
    # 조선비즈
    {"name": "조선비즈",     "url": "https://biz.chosun.com/rssfeeds/economy/",    "source": "chosun"},
    # 연합인포맥스
    {"name": "인포맥스 뉴스","url": "https://news.einfomax.co.kr/rss/allArticle.xml","source": "einfomax"},

    # ── Investing.com ─────────────────────────────────────────
    # 한국어판 — 전체 / 경제 / 주식 / 원자재 / 외환
    {
        "name":   "Investing KR 전체",
        "url":    "https://kr.investing.com/rss/news.rss",
        "source": "investing_kr",
        "cloudflare": False,   # RSS XML 은 대부분 일반 요청 통과
    },
    {
        "name":   "Investing KR 경제",
        "url":    "https://kr.investing.com/rss/news_301.rss",
        "source": "investing_kr",
        "cloudflare": False,
    },
    {
        "name":   "Investing KR 주식",
        "url":    "https://kr.investing.com/rss/news_25.rss",
        "source": "investing_kr",
        "cloudflare": False,
    },
    # 영어판 — 경제 / 외환 / 원자재
    {
        "name":   "Investing EN Economy",
        "url":    "https://www.investing.com/rss/news_301.rss",
        "source": "investing_en",
        "cloudflare": False,
    },
    {
        "name":   "Investing EN Forex",
        "url":    "https://www.investing.com/rss/news_14.rss",
        "source": "investing_en",
        "cloudflare": False,
    },
    {
        "name":   "Investing EN Commodities",
        "url":    "https://www.investing.com/rss/news_8.rss",
        "source": "investing_en",
        "cloudflare": False,
    },
]

# ── BS 목록 크롤링 소스 (기사 목록 페이지) ───────────────────
BS_SOURCES = [
    {
        "name": "한국경제 경제",
        "url": "https://www.hankyung.com/economy",
        "source": "hankyung",
        "list_selector": "div.article-list a, h3.tit a, a.news-tit",
        "title_selector": "h1.headline, h1, .article-header h1",
        "body_selector": "div.article-body, div#articletxt, div.article_body",
        "base_url": "https://www.hankyung.com",
    },
    {
        "name": "매일경제 경제",
        "url": "https://www.mk.co.kr/news/economy/",
        "source": "mk",
        "list_selector": "a.news_item, h3.news_ttl a, div.news_list a",
        "title_selector": "h2.news_ttl, h1.top_title, h2.news_detail_headline",
        "body_selector": "div.news_cnt_detail_wrap, div#article_body, div.art_txt",
        "base_url": "https://www.mk.co.kr",
    },
    {
        "name": "이데일리 경제",
        "url": "https://www.edaily.co.kr/news/newspath/?newsid=economic",
        "source": "edaily",
        "list_selector": "div.newsbox_04 a, h3.sub_title a",
        "title_selector": "h1.news_tit, h1",
        "body_selector": "div.news_body, div#newsBody",
        "base_url": "https://www.edaily.co.kr",
    },
    {
        "name": "머니투데이 증권",
        "url": "https://news.mt.co.kr/mtview.php?category=stockMk",
        "source": "mt",
        "list_selector": "ul.news_list a, div.news_item a, h3 a",
        "title_selector": "h1.tit_view, h1",
        "body_selector": "div#textBody, div.view_text",
        "base_url": "https://news.mt.co.kr",
    },
]


# ══════════════════════════════════════════════════════════════
#  유틸 함수
# ══════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """HTML 태그 및 불필요 공백 제거."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_noise_elements(soup: BeautifulSoup) -> BeautifulSoup:
    """
    기사 본문 추출 전 UI 쓰레기 제거.
    네이버 구독버튼, SNS공유, 기자정보, 저작권, 관련기사 등을 모두 제거.
    """
    # ── 태그 자체 제거 ────────────────────────────────────────
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "iframe", "noscript", "form", "button"]):
        tag.decompose()

    # ── 클래스/ID 키워드로 제거 ───────────────────────────────
    NOISE_KEYWORDS = [
        # 공통 UI
        "share", "social", "sns", "kakao", "twitter", "facebook",
        "subscribe", "subscription", "reporter", "journalist",
        "copyright", "footer", "header", "nav", "navigation",
        "related", "recommend", "popular", "ranking", "more-news",
        "comment", "reply", "ad", "advertisement", "banner",
        "breadcrumb", "pagination", "paging", "tag-list",
        # 네이버 뉴스 전용
        "media_end_linked", "media_end_head_info",    # 기자 정보
        "media_end_head_sns",                          # SNS 공유
        "media_end_head_copyright",
        "media_end_relation",                          # 관련기사
        "end_body_caption",
        "reporter_area", "journalist_card",
        # 연합뉴스 전용
        "story-news__copyright", "btn-story",
        "btn-share", "share-area", "relate-news",
        # 한경/매경
        "article-social", "article-btn", "article-relation",
        # AtozBoard
        "article-view-head-info", "copyright-info",
        "article-photo-info",
    ]

    for keyword in NOISE_KEYWORDS:
        for el in soup.find_all(
            lambda tag, kw=keyword:
                kw in " ".join(tag.get("class", [])).lower() or
                kw in (tag.get("id", "") or "").lower()
        ):
            el.decompose()

    return soup


def _is_duplicate(conn, url: str) -> bool:
    row = conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone()
    return row is not None


def _get_domain(url: str) -> str:
    """URL에서 도메인 추출."""
    m = re.search(r"https?://([^/]+)", url)
    return m.group(1) if m else ""


def _select_first(soup: BeautifulSoup, selectors: List[str], attr: str = None) -> str:
    """
    셀렉터 목록을 순서대로 시도해 처음 히트한 텍스트 반환.
    attr 지정 시 해당 속성값 반환 (eg. og:title content).
    """
    for sel in selectors:
        try:
            el = soup.select_one(sel)
            if el:
                if attr:
                    val = el.get(attr, "").strip()
                    if val:
                        return val
                else:
                    txt = _clean_text(el.get_text())
                    if txt:
                        return txt
        except Exception:
            continue
    return ""


def _fetch_with_retry(url: str, extra_headers: Dict = None,
                      max_retries: int = 3, timeout: int = 12) -> Optional[requests.Response]:
    """
    재시도(최대 3회) + 지수 백오프로 GET 요청.
    실패 시 None 반환.
    """
    headers = dict(HEADERS)
    if extra_headers:
        headers.update(extra_headers)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout,
                                allow_redirects=True)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in (403, 429, 503):
                wait = 2 ** attempt
                log.warning(f"HTTP {resp.status_code} — {attempt}회 시도, {wait}s 대기: {url}")
                time.sleep(wait)
            elif resp.status_code == 404:
                log.warning(f"HTTP 404 (존재하지 않음): {url}")
                return None
            else:
                log.warning(f"HTTP {resp.status_code}: {url}")
                return None
        except requests.exceptions.Timeout:
            log.warning(f"타임아웃 {attempt}/{max_retries}: {url}")
            time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError as e:
            log.warning(f"연결 실패 {attempt}/{max_retries}: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            log.error(f"예상치 못한 오류: {e} — {url}")
            return None

    log.error(f"최대 재시도 초과: {url}")
    return None


def _fetch_cloudflare(url: str, extra_headers: Dict = None,
                      max_retries: int = 3, timeout: int = 20) -> Optional[requests.Response]:
    """
    Cloudflare 보호 사이트 전용 GET 요청.

    우선순위:
      1) cloudscraper 설치 → cloudscraper 세션 사용
      2) 미설치 → 일반 _fetch_with_retry() 로 폴백 (차단될 수 있음)

    반환값: requests.Response 호환 객체 또는 None
    에러 처리:
      - 403/429  : Cloudflare 차단 → 지수 백오프 후 재시도
      - 503      : 일시적 Cloudflare 오류 → 재시도
      - 기타 4xx : 즉시 None 반환
      - 타임아웃  : 재시도
    """
    if not CLOUDSCRAPER_AVAILABLE:
        log.warning(
            "cloudscraper 미설치 — 일반 요청으로 폴백 (차단 가능성 있음). "
            "`pip install cloudscraper` 권장"
        )
        return _fetch_with_retry(url, extra_headers=extra_headers,
                                 max_retries=max_retries, timeout=timeout)

    headers = dict(HEADERS)
    if extra_headers:
        headers.update(extra_headers)

    for attempt in range(1, max_retries + 1):
        try:
            resp = _cloudscraper_session.get(
                url, headers=headers, timeout=timeout, allow_redirects=True
            )

            if resp.status_code == 200:
                return resp

            elif resp.status_code in (403, 429, 503):
                wait = 2 ** attempt
                log.warning(
                    f"[Cloudflare] HTTP {resp.status_code} — "
                    f"{attempt}/{max_retries}회, {wait}s 대기: {url}"
                )
                time.sleep(wait)

            elif resp.status_code == 404:
                log.warning(f"[Cloudflare] HTTP 404: {url}")
                return None

            else:
                log.warning(f"[Cloudflare] HTTP {resp.status_code}: {url}")
                return None

        except requests.exceptions.Timeout:
            log.warning(f"[Cloudflare] 타임아웃 {attempt}/{max_retries}: {url}")
            time.sleep(2 ** attempt)

        except requests.exceptions.ConnectionError as e:
            log.warning(f"[Cloudflare] 연결 실패 {attempt}/{max_retries}: {e}")
            time.sleep(2 ** attempt)

        except Exception as e:
            # cloudscraper 내부 오류 (JS 챌린지 파싱 실패 등) 포함
            log.error(f"[Cloudflare] 예상치 못한 오류 (attempt {attempt}): {e} — {url}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return None

    log.error(f"[Cloudflare] 최대 재시도 초과: {url}")
    return None


def _fetch_auto(url: str, extra_headers: Dict = None,
                cloudflare: bool = False, timeout: int = 12) -> Optional[requests.Response]:
    """
    cloudflare 플래그에 따라 _fetch_cloudflare / _fetch_with_retry 를 자동 선택.
    도메인 셀렉터의 'cloudflare' 키를 읽어 호출하는 단일 진입점.
    """
    if cloudflare:
        return _fetch_cloudflare(url, extra_headers=extra_headers, timeout=timeout)
    return _fetch_with_retry(url, extra_headers=extra_headers, timeout=timeout)


# ══════════════════════════════════════════════════════════════
#  단일 URL 크롤링
# ══════════════════════════════════════════════════════════════

def crawl_single_url(url: str) -> Optional[Dict]:
    """
    기사 URL 하나를 크롤링해 기사 딕셔너리 반환.
    도메인별 전용 셀렉터 → OG 태그 → 범용 폴백 순으로 시도.
    실패 시 None 반환.
    """
    domain = _get_domain(url)

    # 도메인 매칭
    matched = None
    for key, sel in DOMAIN_SELECTORS.items():
        if key in domain:
            matched = sel
            break

    extra_headers = matched.get("extra_headers", {}) if matched else {}
    source        = matched["source"]     if matched else "manual"
    use_cf        = matched.get("cloudflare", False) if matched else False

    # ── HTTP 요청 (Cloudflare 보호 여부 자동 판별) ───────────
    resp = _fetch_auto(url, extra_headers=extra_headers, cloudflare=use_cf)
    if resp is None:
        return None

    # 인코딩 보정
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")
    soup = _strip_noise_elements(soup)   # ← 노이즈 제거 먼저

    # ── 제목 추출 ────────────────────────────────────────────
    title_sels = matched["title"] if matched else []
    title = _select_first(soup, title_sels)

    # 폴백 1: Open Graph
    if not title:
        og = soup.find("meta", property="og:title")
        title = og.get("content", "").strip() if og else ""

    # 폴백 2: <title> 태그
    if not title:
        t = soup.find("title")
        title = _clean_text(t.get_text()) if t else ""

    # 폴백 3: 범용 셀렉터
    if not title:
        title = _select_first(soup, FALLBACK_TITLE_SELS)

    if not title or len(title) < 5:
        log.warning(f"제목 추출 실패: {url}")
        return None

    # ── 본문 추출 ────────────────────────────────────────────
    body_sels = matched["body"] if matched else []
    body = _select_first(soup, body_sels)

    # 폴백: Open Graph description
    if not body:
        og_desc = soup.find("meta", property="og:description")
        body = og_desc.get("content", "").strip() if og_desc else ""

    # 폴백: 범용 셀렉터
    if not body:
        body = _select_first(soup, FALLBACK_BODY_SELS)

    # 본문이 너무 짧으면 경고 (실패는 아님 — 제목만이라도 저장)
    if len(body) < 50:
        log.warning(f"본문이 너무 짧음({len(body)}자), 제목만 저장: {title[:40]}")

    # ── 날짜 추출 ────────────────────────────────────────────
    pub_date = datetime.now().strftime("%Y-%m-%d")
    for meta_sel in [
        {"property": "article:published_time"},
        {"name": "pubdate"},
        {"name": "date"},
        {"itemprop": "datePublished"},
    ]:
        el = soup.find("meta", meta_sel)
        if el:
            raw = el.get("content", "")
            m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
            if m:
                pub_date = m.group(0)
                break

    log.info(f"[OK] {source} | {title[:60]}")
    return {
        "source":       source,
        "title":        title,
        "body":         body[:8000],   # 5000→8000자로 증량
        "url":          url,
        "published_at": pub_date,
    }


# ══════════════════════════════════════════════════════════════
#  DB 저장
# ══════════════════════════════════════════════════════════════

def save_articles(articles: List[Dict]) -> int:
    """기사 리스트를 DB에 저장, 중복 URL은 건너뜀. 저장 수 반환."""
    saved = 0
    with get_db() as conn:
        for art in articles:
            if _is_duplicate(conn, art["url"]):
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
                log.error(f"DB 저장 오류: {e} | {art.get('url','')}")
    return saved


# ══════════════════════════════════════════════════════════════
#  RSS 크롤링
# ══════════════════════════════════════════════════════════════

def _parse_rss_body(link: str, source: str) -> str:
    """
    RSS 링크에서 본문 크롤링. 실패 시 빈 문자열 반환.
    Cloudflare 보호 사이트는 _fetch_cloudflare() 를 통해 요청.
    """
    matched = None
    domain  = _get_domain(link)
    for key, sel in DOMAIN_SELECTORS.items():
        if key in domain:
            matched = sel
            break

    extra_headers = matched.get("extra_headers", {}) if matched else {}
    use_cf        = matched.get("cloudflare", False) if matched else False

    resp = _fetch_auto(link, extra_headers=extra_headers,
                       cloudflare=use_cf, timeout=10)
    if not resp:
        return ""

    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")
    soup = _strip_noise_elements(soup)
    body_sels = matched["body"] if matched else FALLBACK_BODY_SELS
    return _select_first(soup, body_sels)[:8000]


def crawl_rss(max_per_feed: int = 80) -> List[Dict]:
    """
    RSS 피드 전체에서 기사 수집 (피드당 최대 max_per_feed개).

    피드 딕셔너리 선택 키:
      cloudflare (bool, 기본 False) — 개별 기사 본문 요청 시 cloudscraper 사용
    """
    articles = []
    for feed_info in RSS_FEEDS:
        try:
            log.info(f"RSS 수집 시작: {feed_info['name']}")
            feed = feedparser.parse(feed_info["url"])

            if not feed.entries:
                log.warning(f"RSS 항목 없음 (피드 차단 또는 빈 피드): {feed_info['url']}")
                continue

            count = 0
            for entry in feed.entries[:max_per_feed]:
                title = _clean_text(entry.get("title", ""))
                if not title:
                    continue
                link  = entry.get("link", "")
                # ── 날짜 정규화: feedparser의 published_parsed (time_struct) 우선 사용 ──
                pub_parsed = entry.get("published_parsed")
                if pub_parsed:
                    try:
                        pub = datetime(*pub_parsed[:6]).strftime("%Y-%m-%d")
                    except Exception:
                        pub = datetime.now().strftime("%Y-%m-%d")
                else:
                    pub_raw = entry.get("published", "")
                    if pub_raw:
                        # 이미 YYYY-MM-DD 형태인지 확인
                        import re as _re
                        m = _re.search(r'(\d{4}-\d{2}-\d{2})', pub_raw)
                        pub = m.group(1) if m else datetime.now().strftime("%Y-%m-%d")
                    else:
                        pub = datetime.now().strftime("%Y-%m-%d")

                body = _parse_rss_body(link, feed_info["source"])

                articles.append({
                    "source":       feed_info["source"],
                    "title":        title,
                    "body":         body,
                    "url":          link,
                    "published_at": pub,
                })
                count += 1
                time.sleep(0.2)

            log.info(f"  → {count}건 수집")
        except Exception as e:
            log.error(f"RSS 오류 ({feed_info['name']}): {e}")
    return articles


# ══════════════════════════════════════════════════════════════
#  BS 목록 크롤링
# ══════════════════════════════════════════════════════════════

def crawl_bs(max_per_source: int = 40) -> List[Dict]:
    """기사 목록 페이지에서 링크 추출 후 각 기사 크롤링."""
    articles = []
    for src in BS_SOURCES:
        try:
            log.info(f"BS 수집 시작: {src['name']}")
            resp = _fetch_with_retry(src["url"])
            if not resp:
                continue

            resp.encoding = resp.apparent_encoding or "utf-8"
            soup = BeautifulSoup(resp.text, "lxml")
            links = soup.select(src["list_selector"])

            seen_urls: set = set()
            count = 0
            for a_tag in links[:max_per_source * 2]:   # 여유분 확보
                href = a_tag.get("href", "")
                if not href or href in seen_urls:
                    continue
                if not href.startswith("http"):
                    href = src["base_url"] + href
                seen_urls.add(href)

                art_resp = _fetch_with_retry(href, timeout=10)
                if not art_resp:
                    continue

                art_resp.encoding = art_resp.apparent_encoding or "utf-8"
                art_soup = BeautifulSoup(art_resp.text, "lxml")

                title_el = art_soup.select_one(src["title_selector"])
                title = _clean_text(title_el.get_text()) if title_el else ""
                if not title:
                    title = _clean_text(a_tag.get_text())
                if not title or len(title) < 5:
                    continue

                body_el = art_soup.select_one(src["body_selector"])
                body = _clean_text(body_el.get_text()) if body_el else ""

                articles.append({
                    "source":       src["source"],
                    "title":        title,
                    "body":         body[:8000],
                    "url":          href,
                    "published_at": datetime.now().strftime("%Y-%m-%d"),
                })
                count += 1
                time.sleep(0.4)
                if count >= max_per_source:
                    break

            log.info(f"  → {count}건 수집")
        except Exception as e:
            log.error(f"BS 오류 ({src['name']}): {e}")
    return articles


# ══════════════════════════════════════════════════════════════
#  urls.txt 파일 기반 크롤링
# ══════════════════════════════════════════════════════════════

def crawl_from_url_file(filepath: str = URL_FILE_PATH) -> Dict:
    """
    urls.txt 의 URL들을 읽어 크롤링 후 DB에 저장.

    중복 처리:
      1) 파일 내 중복 URL → set()으로 제거
      2) DB에 이미 있는 URL → _is_duplicate()로 건너뜀
    """
    if not os.path.exists(filepath):
        return {"error": f"파일 없음: {filepath}"}

    raw_urls: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            url = line.split("#")[0].strip()
            if url:
                raw_urls.append(url)

    seen: set = set()
    unique_urls: List[str] = []
    for u in raw_urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    file_dup = len(raw_urls) - len(unique_urls)
    log.info(f"[urls.txt] {len(unique_urls)}개 URL (파일내 중복 {file_dup}개 제거)")

    crawled, skipped_db, failed = 0, 0, 0
    articles: List[Dict] = []

    with get_db() as conn:
        for url in unique_urls:
            if _is_duplicate(conn, url):
                log.info(f"  [SKIP-DB] {url}")
                skipped_db += 1
                continue

            art = crawl_single_url(url)
            if art:
                articles.append(art)
                crawled += 1
            else:
                failed += 1
            time.sleep(0.5)

    saved = save_articles(articles)

    summary = {
        "total_in_file":   len(raw_urls),
        "unique_urls":     len(unique_urls),
        "file_duplicates": file_dup,
        "crawled":         crawled,
        "skipped_db":      skipped_db,
        "failed":          failed,
        "newly_saved":     saved,
    }
    log.info(f"urls.txt 완료: {summary}")
    return summary


# ══════════════════════════════════════════════════════════════
#  전체 크롤링 실행
# ══════════════════════════════════════════════════════════════

def run_crawl(rss_max: int = 80, bs_max: int = 40) -> Dict:
    """RSS + BS 전체 크롤링 파이프라인."""
    log.info("=" * 60)
    log.info(f"전체 크롤링 시작: {datetime.now()}")

    rss_articles = crawl_rss(max_per_feed=rss_max)
    log.info(f"RSS 총 {len(rss_articles)}개 수집")

    bs_articles = crawl_bs(max_per_source=bs_max)
    log.info(f"BS 총 {len(bs_articles)}개 수집")

    all_articles = rss_articles + bs_articles
    saved = save_articles(all_articles)

    summary = {
        "total_crawled": len(all_articles),
        "newly_saved":   saved,
        "rss_count":     len(rss_articles),
        "bs_count":      len(bs_articles),
    }
    log.info(f"전체 크롤링 완료: {summary}")
    log.info("=" * 60)
    return summary


# ══════════════════════════════════════════════════════════════
#  직접 실행
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from database import init_db
    init_db()

    if "--urls" in sys.argv:
        result = crawl_from_url_file()
    elif "--rss" in sys.argv:
        result = {"rss": len(crawl_rss())}
    elif "--bs" in sys.argv:
        result = {"bs": len(crawl_bs())}
    else:
        result = run_crawl()

    print(result)
