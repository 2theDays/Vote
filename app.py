"""
선거 전략 인사이트: 예측과 전망
================================
네이버 + 구글(Apify) 트렌드 종합 분석 시스템
"""

import streamlit as st
import google.generativeai as genai
from duckduckgo_search import DDGS
import requests
import time
import json
import random
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Dict, List, Optional, Any, Tuple
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="선거 전략: 예측과 전망",
    layout="wide",
    page_icon="🗳️",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
/* 메인 테마 */
.main { background-color: #0f172a; color: #e2e8f0; }
h1, h2, h3 { color: #f1f5f9; }

/* 버튼 스타일 */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

/* 카드 스타일 */
.candidate-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #334155;
}

/* 메트릭 스타일 */
div[data-testid="metric-container"] {
    background: #1e293b;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #334155;
}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #1e293b;
    border-radius: 8px;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    background: #3b82f6 !important;
    color: white !important;
}

/* 프로그레스 바 */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #334155;
}

/* 배지 스타일 */
.party-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    margin-left: 10px;
}
.trend-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 15px;
    font-size: 0.75em;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 설정 상수
# ==========================================
CONFIG = {
    "MAX_CANDIDATES": 10,
    "TREND_DAYS": 180,
    "PREDICTION_DAYS": 30,
    "MAX_NEWS": 30,
    "TIMEOUT": 30,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 2,
    "PARALLEL_WORKERS": 5,
    "COLORS": ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'],
    "PARTY_COLORS": {
        "더불어민주당": "#1d4ed8",
        "민주당": "#1d4ed8",
        "국민의힘": "#b91c1c",
        "개혁신당": "#f97316",
        "정의당": "#eab308",
        "무소속": "#6b7280",
        "기본": "#4b5563"
    }
}

# ==========================================
# 유틸리티 함수들
# ==========================================
def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"{func.__name__} 실패 ({attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            logger.error(f"{func.__name__} 최종 실패: {last_exception}")
            return None
        return wrapper
    return decorator


def generate_cache_key(data: Any) -> str:
    """캐시 키 생성"""
    return hashlib.md5(str(data).encode()).hexdigest()[:16]


def safe_get_value(df: pd.DataFrame, col: str, idx: int = -1) -> Optional[float]:
    """안전한 데이터프레임 값 추출"""
    if df is None or df.empty or col not in df.columns:
        return None
    try:
        return float(df[col].iloc[idx])
    except (IndexError, ValueError, TypeError):
        return None


def clean_json(text: str) -> Optional[Dict]:
    """JSON 텍스트 파싱"""
    if not text:
        return None
    try:
        text = re.sub(r'```json\s*|```\s*', '', text).strip()
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0).replace('\n', ' '))
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 파싱 실패: {e}")
        return None


def contains_name(text: str, name: str) -> bool:
    """텍스트에 이름 포함 여부 확인"""
    if not text or not name:
        return False
    return name in text or (len(name) >= 2 and name[1:] in text)


def get_party_color(party: str) -> str:
    """정당 색상 반환"""
    if not party:
        return CONFIG["PARTY_COLORS"]["기본"]
    for key, color in CONFIG["PARTY_COLORS"].items():
        if key in party:
            return color
    return CONFIG["PARTY_COLORS"]["기본"]


# ==========================================
# 데이터베이스 관리
# ==========================================
class DatabaseManager:
    """분석 히스토리 및 결과 저장 관리"""

    def __init__(self, db_path: str = "election_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    election_name TEXT NOT NULL,
                    candidates TEXT NOT NULL,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    results TEXT,
                    trends_summary TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS candidate_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    candidate_name TEXT NOT NULL,
                    party TEXT,
                    current_role TEXT,
                    past_career TEXT,
                    poll_est INTEGER,
                    analysis TEXT,
                    sns_sentiment TEXT,
                    keywords TEXT,
                    naver_current REAL,
                    naver_predicted REAL,
                    google_current REAL,
                    google_predicted REAL,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history(id)
                )
            ''')
            conn.commit()

    def save_analysis(self, election: str, candidates: List[str],
                      results: List[Dict], trends: Dict) -> int:
        """분석 결과 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 트렌드 요약 생성
            trends_summary = {}
            for source, df in trends.items():
                if not df.empty:
                    trends_summary[source] = {
                        col: {
                            "current": safe_get_value(df, col),
                            "mean": float(df[col].mean()) if col in df.columns else None
                        }
                        for col in df.columns
                    }

            cursor.execute('''
                INSERT INTO analysis_history (election_name, candidates, results, trends_summary)
                VALUES (?, ?, ?, ?)
            ''', (election, json.dumps(candidates), json.dumps(results), json.dumps(trends_summary)))

            analysis_id = cursor.lastrowid

            for result in results:
                cursor.execute('''
                    INSERT INTO candidate_data
                    (analysis_id, candidate_name, party, current_role, past_career,
                     poll_est, analysis, sns_sentiment, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    result.get('name', ''),
                    result.get('party', ''),
                    result.get('current_role', ''),
                    result.get('past_career', ''),
                    result.get('poll_est', 0),
                    result.get('analysis', ''),
                    result.get('sns_sentiment', ''),
                    json.dumps(result.get('keywords', []))
                ))

            conn.commit()
            return analysis_id

    def get_history(self, limit: int = 10) -> List[Dict]:
        """분석 히스토리 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, election_name, candidates, analysis_date
                FROM analysis_history
                ORDER BY analysis_date DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_analysis_detail(self, analysis_id: int) -> Optional[Dict]:
        """특정 분석 상세 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (analysis_id,))
            history = cursor.fetchone()

            if not history:
                return None

            cursor.execute('SELECT * FROM candidate_data WHERE analysis_id = ?', (analysis_id,))
            candidates = [dict(row) for row in cursor.fetchall()]

            return {
                "history": dict(history),
                "candidates": candidates
            }


# ==========================================
# API 및 모델 관리
# ==========================================
def load_api_keys() -> Dict[str, Optional[str]]:
    """API 키 로드"""
    try:
        keys = {
            "gemini": st.secrets["GEMINI_API_KEY"],
            "naver_id": st.secrets["NAVER_CLIENT_ID"],
            "naver_secret": st.secrets["NAVER_CLIENT_SECRET"],
        }
        try:
            keys["apify"] = st.secrets["APIFY_API_KEY"]
        except KeyError:
            keys["apify"] = None
        return keys
    except Exception as e:
        logger.error(f"API 키 로드 실패: {e}")
        st.error("API 키 로드 실패. `.streamlit/secrets.toml` 파일을 확인하세요.")
        st.stop()


@st.cache_data(ttl=3600)
def get_best_model(api_key: str) -> str:
    """최적의 Gemini 모델 선택"""
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models()
                  if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if 'flash' in m.lower():
                return m
        if models:
            return models[0]
    except Exception as e:
        logger.warning(f"모델 목록 조회 실패: {e}")
    return "models/gemini-1.5-flash"


def validate_candidates(candidates: List[str]) -> Tuple[bool, str]:
    """후보자 목록 유효성 검사"""
    if not candidates:
        return False, "후보자를 최소 1명 입력하세요"
    if len(candidates) > CONFIG["MAX_CANDIDATES"]:
        return False, f"후보자는 최대 {CONFIG['MAX_CANDIDATES']}명까지 가능합니다"
    for c in candidates:
        if len(c) < 2:
            return False, f"'{c}'는 너무 짧습니다 (최소 2자)"
        if len(c) > 20:
            return False, f"'{c}'는 너무 깁니다 (최대 20자)"
    return True, ""


# ==========================================
# 트렌드 데이터 수집
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_naver_trend(candidates: List[str], n_id: str, n_secret: str) -> pd.DataFrame:
    """네이버 데이터랩 트렌드 수집"""
    try:
        url = "https://openapi.naver.com/v1/datalab/search"
        headers = {
            "X-Naver-Client-Id": n_id,
            "X-Naver-Client-Secret": n_secret,
            "Content-Type": "application/json"
        }
        end = datetime.now()
        start = end - timedelta(days=CONFIG["TREND_DAYS"])
        all_data = {}

        for i in range(0, len(candidates), 5):
            batch = candidates[i:i+5]
            body = {
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": end.strftime("%Y-%m-%d"),
                "timeUnit": "week",
                "keywordGroups": [{"groupName": c, "keywords": [c]} for c in batch]
            }

            resp = requests.post(url, headers=headers, json=body, timeout=CONFIG["TIMEOUT"])

            if resp.status_code == 200:
                for item in resp.json().get('results', []):
                    df = pd.DataFrame(item['data'])
                    if not df.empty:
                        df['period'] = pd.to_datetime(df['period'])
                        df.set_index('period', inplace=True)
                        all_data[item['title']] = df['ratio']
            else:
                logger.warning(f"네이버 API 오류: {resp.status_code}")

            time.sleep(0.3)

        return pd.DataFrame(all_data) if all_data else pd.DataFrame()

    except Exception as e:
        logger.error(f"네이버 트렌드 수집 실패: {e}")
        return pd.DataFrame()


def get_google_trend_apify(candidates: List[str], api_key: str,
                           status_callback=None) -> pd.DataFrame:
    """Apify Google Trends 수집"""
    if not api_key:
        return pd.DataFrame()

    try:
        run_url = f"https://api.apify.com/v2/acts/emastra~google-trends-scraper/runs?token={api_key}"
        run_input = {
            "searchTerms": candidates,
            "timeRange": "today 6-m",
            "geo": "KR",
            "isMultiple": True,
            "skipDebugScreen": True,
            "maxItems": 1
        }

        resp = requests.post(run_url, json=run_input, timeout=CONFIG["TIMEOUT"])

        if resp.status_code == 201:
            run_id = resp.json().get("data", {}).get("id")

            if run_id:
                for i in range(45):
                    time.sleep(2)
                    if status_callback:
                        status_callback(f"📊 구글 트렌드 수집 중... ({i*2}/90초)")

                    status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_key}"
                    status_resp = requests.get(status_url, timeout=10)

                    if status_resp.status_code == 200:
                        status = status_resp.json().get("data", {}).get("status")

                        if status == "SUCCEEDED":
                            dataset_id = status_resp.json().get("data", {}).get("defaultDatasetId")
                            if dataset_id:
                                items_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={api_key}"
                                items_resp = requests.get(items_url, timeout=10)

                                if items_resp.status_code == 200:
                                    return _parse_google_trends(items_resp.json(), candidates)
                            break
                        elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                            break

    except Exception as e:
        logger.error(f"구글 트렌드 수집 실패: {e}")

    return pd.DataFrame()


def _parse_google_trends(items: List[Dict], candidates: List[str]) -> pd.DataFrame:
    """구글 트렌드 데이터 파싱"""
    all_data = {}

    for item in items:
        timeline = item.get("interestOverTime", {}).get("timelineData", [])

        if timeline:
            for idx, name in enumerate(candidates):
                dates, values = [], []

                for point in timeline:
                    try:
                        ts = point.get("time")
                        if not ts:
                            continue

                        dt = datetime.fromtimestamp(int(ts))
                        val_list = point.get("value", [])
                        val = val_list[idx] if idx < len(val_list) else None

                        if val is not None:
                            dates.append(dt)
                            values.append(val)
                    except Exception:
                        continue

                if dates and values:
                    all_data[name] = pd.Series(values, index=dates)

    if all_data:
        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    return pd.DataFrame()


def get_news_trend(candidates: List[str]) -> pd.DataFrame:
    """뉴스 언급량 기반 트렌드"""
    ddgs = DDGS()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=CONFIG["TREND_DAYS"])
    all_counts = {}

    for name in candidates:
        try:
            time.sleep(random.uniform(0.3, 0.7))
            news = ddgs.news(f'"{name}"', region="kr-kr", safesearch="off", max_results=100)

            if news:
                date_counts = {}
                for article in news:
                    date_str = article.get('date', '')
                    try:
                        if date_str:
                            dt = pd.to_datetime(date_str).date()
                            if start_date.date() <= dt <= end_date.date():
                                week_start = dt - timedelta(days=dt.weekday())
                                date_counts[week_start] = date_counts.get(week_start, 0) + 1
                    except Exception:
                        continue

                if date_counts:
                    all_counts[name] = date_counts

        except Exception as e:
            logger.warning(f"뉴스 트렌드 수집 실패 ({name}): {e}")

    if not all_counts:
        return pd.DataFrame()

    # 정규화
    global_max = max([max(c.values()) for c in all_counts.values() if c], default=1)
    trend_data = {
        name: {k: (v / global_max) * 100 for k, v in counts.items()}
        for name, counts in all_counts.items()
    }

    # 모든 날짜 수집
    all_dates = set()
    for data in trend_data.values():
        all_dates.update(data.keys())

    if all_dates:
        df_data = {name: [data.get(d, 0) for d in sorted(all_dates)]
                   for name, data in trend_data.items()}
        df = pd.DataFrame(df_data, index=sorted(all_dates))
        df.index = pd.to_datetime(df.index)
        return df

    return pd.DataFrame()


def get_all_trends(candidates: List[str], keys: Dict,
                   status_container) -> Dict[str, pd.DataFrame]:
    """모든 트렌드 데이터 통합 수집"""
    results = {"naver": pd.DataFrame(), "google": pd.DataFrame()}

    # 네이버 트렌드
    status_container.info("📊 네이버 트렌드 수집 중...")
    results["naver"] = get_naver_trend(
        tuple(candidates), keys["naver_id"], keys["naver_secret"]
    )

    # 구글 트렌드 (Apify)
    if keys.get("apify"):
        results["google"] = get_google_trend_apify(
            candidates, keys["apify"],
            lambda msg: status_container.info(msg)
        )

    # Apify 실패 시 뉴스 언급량
    if results["google"].empty:
        status_container.info("📊 뉴스 언급량 수집 중...")
        results["google"] = get_news_trend(candidates)

    return results


# ==========================================
# 후보자 데이터 수집
# ==========================================
def collect_candidate_data(keyword: str, election_name: str) -> Dict[str, Dict]:
    """후보자 관련 모든 데이터 수집 (병렬)"""
    ddgs = DDGS()
    collected = {
        "news": {"text": [], "links": []},
        "sns": {"text": [], "links": []},
        "community": {"text": [], "links": []},
        "wiki": {"text": [], "links": []},
        "youtube": {"text": [], "links": []}
    }

    def _collect_news():
        try:
            time.sleep(random.uniform(0.3, 0.6))
            news = ddgs.news(f'"{keyword}" 2025 OR 2026', region="kr-kr",
                           safesearch="off", max_results=15)
            if not news or len(news) < 3:
                news = ddgs.news(f'"{keyword}" {election_name}', region="kr-kr",
                               safesearch="off", max_results=15)

            for r in (news or []):
                title, body = r.get('title', ''), r.get('body', '')
                if contains_name(title, keyword) or contains_name(body, keyword):
                    date_str = r.get('date', '')
                    collected["news"]["text"].append(f"[{date_str}] {title}: {body[:200]}")
                    collected["news"]["links"].append({
                        "title": title[:60],
                        "url": r.get('url', '#'),
                        "source": r.get('source', ''),
                        "date": date_str,
                        "body": body[:300]
                    })
        except Exception as e:
            logger.warning(f"뉴스 수집 실패: {e}")

    def _collect_wiki():
        try:
            time.sleep(random.uniform(0.3, 0.6))
            wiki = ddgs.text(f'"{keyword}" (site:namu.wiki OR site:ko.wikipedia.org)',
                           region="kr-kr", safesearch="off", max_results=5)
            for r in (wiki or []):
                title, body = r.get('title', ''), r.get('body', '')
                if contains_name(title, keyword) or contains_name(body, keyword):
                    collected["wiki"]["text"].append(f"{title}: {body[:300]}")
                    collected["wiki"]["links"].append({
                        "title": title[:60],
                        "url": r.get('href', '#')
                    })
        except Exception as e:
            logger.warning(f"위키 수집 실패: {e}")

    def _collect_profile():
        try:
            time.sleep(random.uniform(0.3, 0.6))
            profile = ddgs.text(f'"{keyword}" 정당 소속 현재 2025',
                              region="kr-kr", safesearch="off", max_results=5)
            for r in (profile or []):
                title, body = r.get('title', ''), r.get('body', '')
                if contains_name(title, keyword) or contains_name(body, keyword):
                    collected["wiki"]["text"].insert(0, f"{title}: {body[:250]}")
                    collected["wiki"]["links"].insert(0, {
                        "title": title[:60],
                        "url": r.get('href', '#')
                    })
        except Exception as e:
            logger.warning(f"프로필 수집 실패: {e}")

    def _collect_sns():
        try:
            time.sleep(random.uniform(0.3, 0.6))
            sns = ddgs.text(
                f'"{keyword}" (site:blog.naver.com OR site:cafe.naver.com OR site:tistory.com)',
                region="kr-kr", safesearch="off", max_results=15
            )
            for r in (sns or []):
                title, body = r.get('title', ''), r.get('body', '')
                if contains_name(title, keyword) or contains_name(body, keyword):
                    url = r.get('href', '')
                    source = ("네이버블로그" if 'blog.naver' in url else
                             "네이버카페" if 'cafe.naver' in url else
                             "티스토리" if 'tistory' in url else "SNS")
                    collected["sns"]["text"].append(f"[{source}] {title}: {body[:150]}")
                    collected["sns"]["links"].append({
                        "title": title[:60],
                        "url": url,
                        "source": source
                    })
        except Exception as e:
            logger.warning(f"SNS 수집 실패: {e}")

    def _collect_community():
        try:
            time.sleep(random.uniform(0.3, 0.6))
            comm = ddgs.text(f'"{keyword}" (site:dcinside.com OR site:clien.net OR site:fmkorea.com)',
                           region="kr-kr", safesearch="off", max_results=10)
            for r in (comm or []):
                title, body = r.get('title', ''), r.get('body', '')
                if contains_name(title, keyword) or contains_name(body, keyword):
                    url = r.get('href', '')
                    source = ("디시인사이드" if 'dcinside' in url else
                             "클리앙" if 'clien' in url else
                             "FM코리아" if 'fmkorea' in url else "커뮤니티")
                    collected["community"]["text"].append(f"[{source}] {title}: {body[:150]}")
                    collected["community"]["links"].append({
                        "title": title[:60],
                        "url": url,
                        "source": source
                    })
        except Exception as e:
            logger.warning(f"커뮤니티 수집 실패: {e}")

    def _collect_youtube():
        try:
            time.sleep(random.uniform(0.3, 0.6))
            videos = ddgs.videos(f'"{keyword}" {election_name}',
                               region="kr-kr", safesearch="off", max_results=5)
            for r in (videos or []):
                title = r.get('title', '')
                if contains_name(title, keyword):
                    collected["youtube"]["text"].append(title)
                    collected["youtube"]["links"].append({
                        "title": title[:60],
                        "url": r.get('content', '#')
                    })
        except Exception as e:
            logger.warning(f"유튜브 수집 실패: {e}")

    # 병렬 실행
    with ThreadPoolExecutor(max_workers=CONFIG["PARALLEL_WORKERS"]) as executor:
        futures = [
            executor.submit(_collect_news),
            executor.submit(_collect_wiki),
            executor.submit(_collect_profile),
            executor.submit(_collect_sns),
            executor.submit(_collect_community),
            executor.submit(_collect_youtube)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"데이터 수집 작업 실패: {e}")

    return collected


# ==========================================
# AI 분석
# ==========================================
def detect_party(text: str) -> str:
    """텍스트에서 정당 감지"""
    party_patterns = {
        "더불어민주당": ["더불어민주당", "민주당 소속", "민주당 공천", "민주당원"],
        "국민의힘": ["국민의힘", "국민의힘 소속", "국민의힘 공천"],
        "개혁신당": ["개혁신당"],
        "정의당": ["정의당"],
        "무소속": ["무소속"]
    }

    max_matches = 0
    detected = "정보 부족"

    for party, patterns in party_patterns.items():
        matches = sum(text.count(p) for p in patterns)
        if matches > max_matches:
            max_matches = matches
            detected = party

    return detected


def extract_career_timeline(news_links: List[Dict]) -> List[str]:
    """뉴스에서 경력 타임라인 추출"""
    timeline = []
    governments = [
        (pd.Timestamp('2013-02-25'), "이명박 정부"),
        (pd.Timestamp('2017-05-10'), "박근혜 정부"),
        (pd.Timestamp('2022-05-10'), "문재인 정부"),
        (pd.Timestamp('2025-05-10'), "윤석열 정부"),
    ]

    for link in news_links:
        date_str = link.get('date', '')
        if not date_str:
            continue

        try:
            article_date = pd.to_datetime(date_str)
            content = f"{link.get('title', '')} {link.get('body', '')}"

            # 정부 시기 판별
            gov = "현 정부"
            for cutoff, name in governments:
                if article_date < cutoff:
                    gov = name
                    break

            # 경력 추출
            careers = []
            if "청와대" in content or "대통령실" in content:
                if "수석" in content:
                    careers.append(f"{gov} 수석")
                else:
                    careers.append(f"{gov} 청와대" if "이명박" in gov or "박근혜" in gov or "문재인" in gov else f"{gov} 대통령실")

            if "국회의원" in content:
                for term in ["22대", "21대", "20대", "19대", "18대"]:
                    if term in content:
                        careers.append(f"{term} 국회의원")
                        break

            if "도지사" in content and "후보" not in content:
                careers.append("광역단체장")
            if "장관" in content:
                careers.append("장관")

            timeline.extend(careers)

        except Exception:
            continue

    # 중복 제거
    seen = set()
    unique = []
    for item in timeline:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique[:5]


def analyze_candidate(model, name: str, collected: Dict,
                      trend_info: str) -> Dict:
    """후보자 AI 분석"""
    news_cnt = len(collected.get("news", {}).get("links", []))
    sns_cnt = len(collected.get("sns", {}).get("links", []))
    community_cnt = len(collected.get("community", {}).get("links", []))

    # 경력 추출
    career_timeline = extract_career_timeline(
        collected.get("news", {}).get("links", [])
    )

    # 텍스트 병합
    all_text = (
        collected.get("news", {}).get("text", [])[:5] +
        collected.get("wiki", {}).get("text", [])[:5] +
        collected.get("sns", {}).get("text", [])[:3] +
        collected.get("community", {}).get("text", [])[:2]
    )
    raw_text = "\n".join(all_text)

    # 정당 감지
    detected_party = detect_party(raw_text)

    # 화제성 점수 계산
    poll_est = 50
    try:
        if trend_info != "데이터 없음":
            match = re.search(r'현재\s*([\d.]+)', trend_info)
            if match:
                poll_est = min(int(float(match.group(1)) * 8), 100)
    except Exception:
        pass

    # 데이터 부족 시 기본값 반환
    if not raw_text or len(raw_text) < 30:
        return {
            "name": name,
            "party": detected_party,
            "current_role": "정보 부족",
            "past_career": ", ".join(career_timeline) if career_timeline else "정보 부족",
            "poll_est": poll_est,
            "analysis": "수집된 데이터가 부족합니다.",
            "sns_sentiment": "분석 불가",
            "keywords": [name]
        }

    career_info = ", ".join(career_timeline) if career_timeline else "경력 정보 없음"

    prompt = f'''다음은 {name} 후보 정보입니다.

트렌드: {trend_info}
추정 경력: {career_info}

수집된 데이터:
{raw_text[:2500]}

위 정보를 바탕으로 JSON 형식으로 분석 결과를 작성하세요:
{{"name":"{name}","party":"정당명(더불어민주당/국민의힘/개혁신당/무소속 중 하나)","current_role":"현재 직함","past_career":"주요 경력 요약","poll_est":{poll_est},"analysis":"뉴스 기반 분석 요약","sns_sentiment":"SNS 여론 분석","keywords":["관련 키워드 3-5개"]}}'''

    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            resp = model.generate_content(prompt)
            result = clean_json(resp.text)

            if result and 'name' in result:
                # 수치 정규화
                if isinstance(result.get('poll_est'), str):
                    nums = re.findall(r'\d+', str(result['poll_est']))
                    result['poll_est'] = int(nums[0]) if nums else poll_est

                # 정당 보완
                if not result.get('party') or result['party'] in ['정보 없음', '']:
                    result['party'] = detected_party

                # 경력 보완
                if not result.get('past_career') or result['past_career'] == '정보 없음':
                    result['past_career'] = ", ".join(career_timeline) if career_timeline else "정보 부족"

                return result

        except Exception as e:
            logger.warning(f"AI 분석 실패 ({attempt + 1}/{CONFIG['MAX_RETRIES']}): {e}")
            time.sleep(CONFIG["RETRY_DELAY"] * (attempt + 1))

    # 폴백 결과
    return {
        "name": name,
        "party": detected_party,
        "current_role": "정보 확인 필요",
        "past_career": ", ".join(career_timeline) if career_timeline else "정보 부족",
        "poll_est": poll_est,
        "analysis": f"{name} 후보: 뉴스 {news_cnt}건, SNS {sns_cnt}건, 커뮤니티 {community_cnt}건 분석",
        "sns_sentiment": f"SNS/커뮤니티 {sns_cnt + community_cnt}건 수집",
        "keywords": [name, detected_party] if detected_party != "정보 부족" else [name]
    }


# ==========================================
# 예측
# ==========================================
def predict_future(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """미래 트렌드 예측 (선형 회귀)"""
    if df.empty:
        return pd.DataFrame()

    future_df = pd.DataFrame()
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]

    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 5:
            continue

        recent = series.tail(min(30, len(series)))
        x = np.arange(len(recent))
        y = recent.values

        if len(x) > 1:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                predictions = p(np.arange(len(recent), len(recent) + days))
                min_val = max(recent.iloc[-1] * 0.3, 1)
                future_df[col] = np.clip(predictions, min_val, 100)
            except Exception as e:
                logger.warning(f"예측 실패 ({col}): {e}")

    if not future_df.empty:
        future_df.index = future_dates

    return future_df


# ==========================================
# 시각화
# ==========================================
def create_trend_chart(trends: Dict[str, pd.DataFrame],
                       pred_trends: Dict[str, pd.DataFrame],
                       candidates: List[str],
                       google_source: str) -> go.Figure:
    """트렌드 비교 차트 생성"""
    colors = CONFIG["COLORS"]
    naver_df = trends.get("naver", pd.DataFrame())
    google_df = trends.get("google", pd.DataFrame())
    pred_naver = pred_trends.get("naver", pd.DataFrame())
    pred_google = pred_trends.get("google", pd.DataFrame())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("🟢 네이버 트렌드", f"🔵 {google_source}"),
        horizontal_spacing=0.08
    )

    # 네이버 차트
    if not naver_df.empty:
        for idx, col in enumerate(candidates):
            if col in naver_df.columns:
                color = colors[idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=naver_df.index, y=naver_df[col],
                        mode='lines', name=col,
                        line=dict(color=color, width=2.5),
                        legendgroup=col
                    ),
                    row=1, col=1
                )
                if not pred_naver.empty and col in pred_naver.columns:
                    pred_x = [naver_df.index[-1]] + list(pred_naver.index)
                    pred_y = [naver_df[col].iloc[-1]] + list(pred_naver[col])
                    fig.add_trace(
                        go.Scatter(
                            x=pred_x, y=pred_y, mode='lines',
                            line=dict(color=color, width=2.5, dash='dot'),
                            legendgroup=col, showlegend=False,
                            hovertemplate='예측: %{y:.1f}<extra></extra>'
                        ),
                        row=1, col=1
                    )

    # 구글/뉴스 차트
    if not google_df.empty:
        for idx, col in enumerate(candidates):
            if col in google_df.columns:
                color = colors[idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=google_df.index, y=google_df[col],
                        mode='lines',
                        line=dict(color=color, width=2.5),
                        legendgroup=col, showlegend=False
                    ),
                    row=1, col=2
                )
                if not pred_google.empty and col in pred_google.columns:
                    pred_x = [google_df.index[-1]] + list(pred_google.index)
                    pred_y = [google_df[col].iloc[-1]] + list(pred_google[col])
                    fig.add_trace(
                        go.Scatter(
                            x=pred_x, y=pred_y, mode='lines',
                            line=dict(color=color, width=2.5, dash='dot'),
                            legendgroup=col, showlegend=False
                        ),
                        row=1, col=2
                    )
    else:
        fig.add_annotation(
            text="데이터 없음", xref="x2", yref="y2",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#94a3b8")
        )

    fig.update_layout(
        height=420,
        template="plotly_dark",
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font=dict(color='#e2e8f0', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=50, r=50, t=60, b=100),
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155', tickangle=-45)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155', title_text="검색 지수")

    return fig


def create_radar_chart(results: List[Dict], trends: Dict) -> go.Figure:
    """후보자 비교 레이더 차트"""
    categories = ['화제성', '뉴스 노출', 'SNS 반응', '검색 트렌드', '종합 점수']

    fig = go.Figure()
    colors = CONFIG["COLORS"]

    naver_df = trends.get("naver", pd.DataFrame())

    for idx, candidate in enumerate(results):
        name = candidate.get('name', '')

        # 점수 계산
        poll_est = candidate.get('poll_est', 50)

        naver_score = 50
        if not naver_df.empty and name in naver_df.columns:
            naver_score = min(float(naver_df[name].mean()), 100)

        # 가상 점수 (실제로는 더 정교한 계산 필요)
        news_score = min(poll_est * 1.1, 100)
        sns_score = min(poll_est * 0.9, 100)
        total_score = (poll_est + naver_score + news_score + sns_score) / 4

        values = [poll_est, news_score, sns_score, naver_score, total_score]
        values.append(values[0])  # 닫힌 형태

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=name,
            line=dict(color=colors[idx % len(colors)], width=2),
            fillcolor=f"rgba{tuple(list(int(colors[idx % len(colors)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#334155'
            ),
            angularaxis=dict(gridcolor='#334155'),
            bgcolor='#1e293b'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.2,
            xanchor="center", x=0.5
        ),
        paper_bgcolor='#0f172a',
        font=dict(color='#e2e8f0'),
        height=400,
        margin=dict(l=80, r=80, t=40, b=80)
    )

    return fig


def create_comparison_bar(results: List[Dict], trends: Dict) -> go.Figure:
    """후보자 비교 막대 차트"""
    names = [r['name'] for r in results]
    poll_scores = [r.get('poll_est', 0) for r in results]

    naver_df = trends.get("naver", pd.DataFrame())
    naver_scores = []
    for name in names:
        if not naver_df.empty and name in naver_df.columns:
            naver_scores.append(float(naver_df[name].iloc[-1]))
        else:
            naver_scores.append(0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='화제성 점수',
        x=names,
        y=poll_scores,
        marker_color='#3b82f6'
    ))

    fig.add_trace(go.Bar(
        name='네이버 트렌드',
        x=names,
        y=naver_scores,
        marker_color='#22c55e'
    ))

    fig.update_layout(
        barmode='group',
        template="plotly_dark",
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font=dict(color='#e2e8f0'),
        height=350,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        margin=dict(l=50, r=50, t=60, b=50)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#334155')

    return fig


# ==========================================
# UI 컴포넌트
# ==========================================
def render_candidate_card(candidate: Dict, collected: Dict,
                          trends: Dict, pred_trends: Dict,
                          google_label: str):
    """후보자 카드 렌더링"""
    name = candidate.get('name', '미상')
    party = candidate.get('party', '정보 없음')
    party_color = get_party_color(party)

    naver_df = trends.get("naver", pd.DataFrame())
    google_df = trends.get("google", pd.DataFrame())
    pred_naver = pred_trends.get("naver", pd.DataFrame())
    pred_google = pred_trends.get("google", pd.DataFrame())

    # 트렌드 배지
    curr_val = safe_get_value(naver_df, name)
    fut_val = safe_get_value(pred_naver, name)

    badge = ""
    badge_color = "#6b7280"
    if curr_val is not None and fut_val is not None:
        if fut_val > curr_val * 1.1:
            badge = "📈 상승예측"
            badge_color = "#22c55e"
        elif fut_val < curr_val * 0.9:
            badge = "📉 하락예측"
            badge_color = "#ef4444"
        else:
            badge = "➡️ 유지"
            badge_color = "#f59e0b"

    # 카드 헤더
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            badge_html = f"<span style='background:{badge_color};color:white;padding:4px 10px;border-radius:15px;font-size:0.7em;margin-left:10px;'>{badge}</span>" if badge else ""
            st.markdown(
                f"### {name} <span style='background:{party_color};color:white;padding:5px 14px;border-radius:8px;font-size:0.5em;margin-left:12px;'>{party}</span>{badge_html}",
                unsafe_allow_html=True
            )
        with col2:
            st.metric("🔥 화제성", f"{candidate.get('poll_est', 0)}")

        # 트렌드 메트릭
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            val = safe_get_value(naver_df, name)
            st.metric("네이버 현재", f"{val:.1f}" if val else "-")
        with m2:
            val = safe_get_value(pred_naver, name)
            st.metric("네이버 예측", f"{val:.1f}" if val else "-")
        with m3:
            val = safe_get_value(google_df, name)
            st.metric(f"{google_label} 현재", f"{val:.1f}" if val else "-")
        with m4:
            val = safe_get_value(pred_google, name)
            st.metric(f"{google_label} 예측", f"{val:.1f}" if val else "-")

        # 프로필 정보
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔷 현재 직함**")
            st.info(candidate.get('current_role', '정보 없음'))
        with c2:
            st.markdown("**⚪ 주요 경력**")
            st.info(candidate.get('past_career', '정보 없음'))

        # 분석 결과
        st.markdown("**📰 뉴스 분석**")
        st.write(candidate.get('analysis', ''))

        # SNS 여론
        st.markdown("**💬 SNS/커뮤니티 여론**")
        sns_sent = str(candidate.get('sns_sentiment', ''))
        if "긍정" in sns_sent:
            st.success(f"🟢 {sns_sent}")
        elif "부정" in sns_sent:
            st.error(f"🔴 {sns_sent}")
        else:
            st.warning(f"🟡 {sns_sent}")

        # 키워드
        keywords = candidate.get('keywords', [])
        if keywords:
            st.markdown(" ".join([f"`#{k}`" for k in keywords[:6]]))

        # 데이터 출처
        st.markdown("**📚 데이터 출처**")
        tabs = st.tabs([
            f"📰 뉴스 ({len(collected.get('news', {}).get('links', []))})",
            f"💬 SNS ({len(collected.get('sns', {}).get('links', []))})",
            f"👥 커뮤니티 ({len(collected.get('community', {}).get('links', []))})",
            f"📖 위키 ({len(collected.get('wiki', {}).get('links', []))})",
            f"📺 유튜브 ({len(collected.get('youtube', {}).get('links', []))})"
        ])

        sources = ['news', 'sns', 'community', 'wiki', 'youtube']
        for tab, src in zip(tabs, sources):
            with tab:
                links = collected.get(src, {}).get('links', [])
                if links:
                    for link in links[:10]:
                        st.markdown(f"- [{link.get('title', '제목 없음')}]({link.get('url', '#')})")
                else:
                    st.caption("수집된 데이터 없음")

        st.divider()


def render_history_section(db: DatabaseManager):
    """분석 히스토리 섹션"""
    history = db.get_history(limit=5)

    if history:
        st.markdown("**📜 최근 분석 기록**")
        for item in history:
            candidates = json.loads(item['candidates'])
            with st.expander(f"📅 {item['analysis_date'][:16]} - {item['election_name']}"):
                st.write(f"후보자: {', '.join(candidates)}")
                if st.button("상세 보기", key=f"hist_{item['id']}"):
                    detail = db.get_analysis_detail(item['id'])
                    if detail:
                        st.json(detail)
    else:
        st.caption("분석 기록이 없습니다.")


def export_results(results: List[Dict], trends: Dict,
                   election: str) -> str:
    """분석 결과 JSON 내보내기"""
    export_data = {
        "election": election,
        "analysis_date": datetime.now().isoformat(),
        "candidates": results,
        "trends_summary": {}
    }

    for source, df in trends.items():
        if not df.empty:
            export_data["trends_summary"][source] = {
                col: {
                    "current": safe_get_value(df, col),
                    "mean": float(df[col].mean()) if col in df.columns else None,
                    "max": float(df[col].max()) if col in df.columns else None
                }
                for col in df.columns
            }

    return json.dumps(export_data, ensure_ascii=False, indent=2)


# ==========================================
# 메인 애플리케이션
# ==========================================
def main():
    st.title("🗳️ 선거 전략 인사이트: 예측과 전망")
    st.caption("네이버 + 구글(Apify) 트렌드 종합 분석 시스템 v2.0")

    # API 키 로드
    keys = load_api_keys()

    # 데이터베이스 초기화
    db = DatabaseManager()

    # 사이드바
    with st.sidebar:
        st.markdown("### ⚙️ 설정")

        # 시스템 상태
        status_items = []
        status_items.append("✅ Gemini API 연결됨")
        status_items.append("✅ 네이버 API 연결됨")
        if keys.get("apify"):
            status_items.append("✅ Apify 연결됨")
        else:
            status_items.append("⚠️ Apify 미설정 (뉴스 언급량 사용)")

        for item in status_items:
            st.markdown(f"<small>{item}</small>", unsafe_allow_html=True)

        st.divider()

        # 입력 설정
        election = st.text_input(
            "📋 분석 대상 선거",
            value="2026년 충청북도지사 선거",
            help="분석할 선거명을 입력하세요"
        )

        st.markdown("**👥 후보자 목록** (줄바꿈 구분)")
        candidates_text = st.text_area(
            "",
            value="신용한\n노영민\n송기섭",
            height=150,
            label_visibility="collapsed",
            help="각 줄에 후보자 이름 하나씩 입력"
        )

        candidates = [c.strip() for c in candidates_text.split('\n') if c.strip()]

        # 유효성 검사
        is_valid, error_msg = validate_candidates(candidates)
        if not is_valid:
            st.error(error_msg)
        else:
            st.success(f"✓ 등록: {len(candidates)}명")

        st.divider()

        # 분석 버튼
        start_analysis = st.button(
            "🚀 종합 분석 실행",
            type="primary",
            use_container_width=True,
            disabled=not is_valid
        )

        st.divider()

        # 히스토리
        with st.expander("📜 분석 히스토리"):
            render_history_section(db)

    # 메인 컨텐츠
    if start_analysis:
        model = genai.GenerativeModel(get_best_model(keys["gemini"]))

        status_container = st.empty()
        progress_bar = st.progress(0)

        try:
            # 1. 트렌드 수집
            status_container.info("📊 트렌드 데이터 수집 중...")
            trends = get_all_trends(candidates, keys, status_container)
            progress_bar.progress(0.2)

            google_source = "구글 트렌드" if keys.get("apify") and not trends["google"].empty else "뉴스 언급량"
            google_label = "구글" if "구글" in google_source else "뉴스"

            # 2. 예측 계산
            status_container.info("🔮 미래 트렌드 예측 중...")
            pred_trends = {
                "naver": predict_future(trends["naver"], CONFIG["PREDICTION_DAYS"]),
                "google": predict_future(trends["google"], CONFIG["PREDICTION_DAYS"])
            }
            progress_bar.progress(0.3)

            # 3. 후보자별 분석
            results = []
            all_collected = {}

            for i, name in enumerate(candidates):
                status_container.info(f"⚡ [{i+1}/{len(candidates)}] {name} 분석 중...")

                # 데이터 수집
                collected = collect_candidate_data(name, election)
                all_collected[name] = collected

                # 트렌드 정보 구성
                curr = safe_get_value(trends.get("naver", pd.DataFrame()), name)
                fut = safe_get_value(pred_trends.get("naver", pd.DataFrame()), name)

                trend_info = "데이터 없음"
                if curr is not None:
                    fut = fut if fut is not None else curr
                    trend_info = f"현재 {curr:.1f}, 예측 {fut:.1f}"

                # AI 분석
                result = analyze_candidate(model, name, collected, trend_info)
                results.append(result)

                progress_bar.progress(0.3 + (0.6 * (i + 1) / len(candidates)))
                time.sleep(1.5)

            # 4. 결과 저장
            status_container.info("💾 분석 결과 저장 중...")
            analysis_id = db.save_analysis(election, candidates, results, trends)

            progress_bar.progress(1.0)
            status_container.empty()
            progress_bar.empty()

            st.success(f"✅ 분석 완료! (ID: {analysis_id})")

            # 결과 표시
            # 탭 구성
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 트렌드 분석", "🎯 후보 비교", "📋 상세 리포트", "💾 데이터 내보내기"
            ])

            with tab1:
                st.subheader("📈 트렌드 예측 시뮬레이션")
                st.caption("실선: 실제 데이터 | 점선: 30일 예측")
                st.plotly_chart(
                    create_trend_chart(trends, pred_trends, candidates, google_source),
                    use_container_width=True
                )

                # 요약 테이블
                st.markdown("**📊 트렌드 수치 요약**")
                summary_data = []
                for name in candidates:
                    row = {"후보": name}
                    row["네이버 현재"] = f"{safe_get_value(trends['naver'], name):.1f}" if safe_get_value(trends['naver'], name) else "-"
                    row["네이버 예측"] = f"{safe_get_value(pred_trends['naver'], name):.1f}" if safe_get_value(pred_trends['naver'], name) else "-"
                    row[f"{google_label} 현재"] = f"{safe_get_value(trends['google'], name):.1f}" if safe_get_value(trends['google'], name) else "-"
                    row[f"{google_label} 예측"] = f"{safe_get_value(pred_trends['google'], name):.1f}" if safe_get_value(pred_trends['google'], name) else "-"
                    summary_data.append(row)

                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True
                )

            with tab2:
                st.subheader("🎯 후보자 비교 분석")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**레이더 차트**")
                    st.plotly_chart(
                        create_radar_chart(results, trends),
                        use_container_width=True
                    )

                with col2:
                    st.markdown("**점수 비교**")
                    st.plotly_chart(
                        create_comparison_bar(results, trends),
                        use_container_width=True
                    )

                # 순위표
                st.markdown("**🏆 화제성 순위**")
                sorted_results = sorted(results, key=lambda x: x.get('poll_est', 0), reverse=True)
                for i, r in enumerate(sorted_results):
                    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                    st.markdown(f"{medal} **{r['name']}** ({r.get('party', '정보 없음')}) - 화제성: {r.get('poll_est', 0)}")

            with tab3:
                st.subheader("📋 후보자별 심층 리포트")
                for result in results:
                    render_candidate_card(
                        result,
                        all_collected.get(result['name'], {}),
                        trends,
                        pred_trends,
                        google_label
                    )

            with tab4:
                st.subheader("💾 분석 결과 내보내기")

                export_json = export_results(results, trends, election)

                st.download_button(
                    label="📥 JSON 다운로드",
                    data=export_json,
                    file_name=f"election_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

                with st.expander("JSON 미리보기"):
                    st.code(export_json, language="json")

        except Exception as e:
            status_container.empty()
            progress_bar.empty()
            st.error(f"❌ 분석 중 오류 발생: {str(e)}")
            logger.error(f"분석 오류: {e}", exc_info=True)

    else:
        # 시작 전 안내
        st.info("👈 사이드바에서 선거와 후보자를 설정한 후 '종합 분석 실행' 버튼을 클릭하세요.")

        # 기능 소개
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 📊 트렌드 분석
            - 네이버 데이터랩
            - 구글 트렌드 (Apify)
            - 뉴스 언급량
            - 30일 미래 예측
            """)
        with col2:
            st.markdown("""
            ### 🤖 AI 분석
            - Gemini 기반 분석
            - 정당/경력 자동 감지
            - SNS 여론 분석
            - 키워드 추출
            """)
        with col3:
            st.markdown("""
            ### 📈 시각화
            - 트렌드 비교 차트
            - 레이더 차트
            - 순위 비교
            - 결과 내보내기
            """)


if __name__ == "__main__":
    main()
