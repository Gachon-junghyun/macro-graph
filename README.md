# macro-graph

뉴스를 수집해 Gemini로 인과관계를 추출하고,  
“전쟁이 나면 유가/금리/공급망이 어떻게 연결되나?”를  
그래프 + 인과 탐색기로 시각화하는 매크로 분석 웹앱입니다. [cite:3]

---

## 주요 기능

- 한국 경제/정치/지정학 관련 뉴스 자동 크롤링 (연합뉴스, 한경, 매경 등) [cite:3]
- Gemini를 이용한 인과관계(“A → B”) 추출 및 인과 체인 생성 [cite:3]
- 공동출현/IDF 기반 매크로 지식 그래프 구축 및 시각화 [cite:3]
- 인과 탐색기 UI를 통한 시나리오 분석  
  - “전쟁”, “금리 인상”, “유가 급등” 같은 이벤트 검색  
  - 순방향/역방향 영향 트리 조회 [cite:3]
- 가격 데이터(yfinance, pykrx)를 함께 불러와 이벤트–자산가격 연결 분석 [cite:3]

---

## 디렉토리 구조

```text
macro-graph/
├── backend/
│   ├── api.py                  # FastAPI 서버 — 모든 엔드포인트
│   ├── causal_routes.py        # 인과 탐색기 전용 라우터
│   ├── historical_crawler.py   # 과거 기사 수집 크롤러
│   ├── crawler.py              # 뉴스 크롤러 (RSS 22개 + BS 4개 + URL파일)
│   ├── run_causal.py           # 인과 체인 추출 실행기 (일일 한도 관리)
│   ├── causal_extractor.py     # Gemini 인과관계 추출 엔진
│   ├── noun_extractor.py       # 명사 추출 + STOPWORDS + Gemini 노이즈 필터
│   ├── graph_builder.py        # 공동출현 그래프 + 멀티홉 경로 탐색
│   ├── concept_normalizer.py   # 인과 노드 정규화 (유사어 통합)
│   ├── database.py             # SQLite 스키마 + 연결 관리
│   ├── price_engine.py         # yfinance/pykrx 가격 수집
│   ├── ingest_news_folder.py   # news/ 폴더 txt 파일 수동 입력
│   ├── macro_graph.db          # SQLite DB (자동 생성, 약 47MB)
│   ├── crawler.log             # 크롤링 로그
│   ├── urls.txt                # 수동 크롤링할 URL 목록
│   └── news/                   # 수동 입력 기사 txt 파일 폴더
└── frontend/
    └── dist/
        ├── index.html              # D3.js Force Graph 메인 그래프 UI
        └── causal_explorer.html    # 인과 탐색기 UI
```
[cite:3]

---

## 설치

Python 3.10+ 권장. [cite:3]

```bash
git clone https://github.com/Gachon-junghyun/macro-graph.git
cd macro-graph/backend

pip install fastapi uvicorn kiwipiepy networkx \
    beautifulsoup4 lxml feedparser requests \
    pykrx yfinance google-genai python-dotenv
```

환경 변수로 Gemini API 키를 설정합니다. [cite:3]

```bash
export GEMINI_API_KEY=your_key_here
```

(원한다면 `api.py` 내에서 하드코딩도 가능하지만 보안상 비추천입니다.) [cite:3]

---

## 서버 실행

```bash
cd macro-graph/backend
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

브라우저에서 접속: [cite:3]

- 메인 그래프: `http://localhost:8000`
- 인과 탐색기: `http://localhost:8000/causal_explorer.html`

---

## 일일 루틴 (운영 플로우)

매일 장 마감 후(권장: 21시 이후), 다음 순서로 실행합니다. [cite:3]

1. **뉴스 크롤링** (약 5–10분)

   ```bash
   python crawler.py
   ```

2. **인과 체인 추출 (Gemini)** (약 58분, 핵심 단계)

   ```bash
   python run_causal.py
   ```

   - Gemini 무료 한도: 하루 250회  
   - `run_causal.py`가 일일 호출 한도와 속도 제한을 자동 관리  
   - 중간에 종료(Ctrl+C)해도 다음 실행 시 이어서 진행  
   - 이미 처리한 기사는 자동으로 건너뜀 [cite:3]

3. **공동출현 그래프 재빌드** (약 1–2분)

   ```bash
   python graph_builder.py
   ```

또는 API로 한 번에: [cite:3]

- `POST /api/pipeline/full`
- `POST /api/pipeline/extract-causality`

---

## 주요 API 엔드포인트

### 인과 탐색 / 시나리오 분석

- 시나리오 분석 — “X가 발생하면 무엇이 영향받나?” [cite:3]

  ```http
  GET /api/causal/scenario?q=전쟁&depth=3
  GET /api/causal/scenario?q=유가+급등&depth=4
  GET /api/causal/impact/중동전쟁?depth=4
  ```

- 경로 탐색 — “A가 B에 어떻게 연결되나?” [cite:3]

  ```http
  GET /api/causal/path?node_a=후티&node_b=인플레
  ```

- 인과 엣지 전문 검색 (그래프 우회) [cite:3]

  ```http
  GET /api/causal/search?q=반도체&category=기술산업
  GET /api/causal/search?q=유가&min_strength=2
  ```

- 카테고리별 체인 조회 [cite:3]

  ```http
  GET /api/causal/categories
  GET /api/causal/chains/지정학
  GET /api/causal/top-chains
  GET /api/causal/chains/search?q=전쟁&category=안보
  ```

- DB 통계 [cite:3]

  ```http
  GET /api/causal/explorer/stats
  GET /api/stats
  ```

### 가격 데이터/파이프라인 관련

- 과거 기사 수집 (예시) [cite:3]

  ```bash
  python historical_crawler.py --days 7 --keyword "금리"
  python historical_crawler.py --days 7 --sources yna_archive,mk_archive
  python historical_crawler.py --list-sources
  python historical_crawler.py --from 2026-03-01 --to 2026-03-15
  ```

- 가격 데이터 수집 및 스파이크 탐지 [cite:3]

  ```http
  POST /api/pipeline/fetch-prices
  POST /api/pipeline/detect-spikes
  ```

- 인과 노드 정규화 (유사어 통합) [cite:3]

  ```http
  POST /api/pipeline/extract
  ```

---

## 현재 DB 스냅샷 (예시)

2026-03-16 기준 DB 통계: [cite:3]

- 수집 기사: 3,182건
- 인과 엣지: 1,903건
- 인과 체인: 776건
- 인과 고유 노드: 2,277개
- 그래프 노드: 40개 (IDF 60% 필터 이후)
- 가격 데이터: 975건
- DB 크기: 47MB

---

## 핵심 설정 위치

- **Gemini API 키**: 환경변수 `GEMINI_API_KEY` 또는 `api.py` 상단 [cite:3]
- **일일 한도 조정**: `run_causal.py` 의 `DAILY_LIMIT`, `RATE_LIMIT_SEC` [cite:3]
- **STOPWORDS / FINANCE_KEYWORDS**: `noun_extractor.py` 내 상수 [cite:3]
- **자산 노드 목록**: `price_engine.py` 의 `ASSET_NODES` [cite:3]
- **IDF 상한 (노이즈 필터)**: `graph_builder.py` 의 `MAX_IDF_RATIO` [cite:3]
- **과거 기사 소스 추가**: `historical_crawler.py` 의 `SEARCH_SOURCES` [cite:3]

---

## 알려진 제한사항

- 네이버 뉴스는 VM/서버 환경에서 봇 차단 이슈가 있어,  
  필요 시 `news/` 폴더에 txt 파일로 수동 입력해야 합니다. [cite:3]
- 인과 노드 정규화가 완전하지 않아  
  “삼성전자 매력 상승” / “삼성전자 관심” 같은 표현이 다른 노드로 분리될 수 있습니다. [cite:3]
- 그래프 노드 수가 너무 적으면 `MAX_IDF_RATIO`를 올려야 합니다. [cite:3]

---

## 향후 개선 아이디어

- 인과 노드 정규화 파이프라인 완전 통합 [cite:3]
- 미국 뉴스 소스(Reuters, CNBC, MarketWatch 등) 추가 [cite:3]
- TF-IDF/Burst 기반 노드 중요도 스코어링 [cite:3]
- D3 기반 방향성 그래프(DAG) 시각화 강화 [cite:3]
- 일일 파이프라인 크론/스케줄러 자동화 [cite:3]

---

## 라이선스

이 레포지토리의 라이선스는 `LICENSE` 파일을 참고하세요. [cite:1]
