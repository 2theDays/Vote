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
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(page_title="선거 전략: 예측과 전망", layout="wide", page_icon="📡")

# CSS 스타일
st.markdown("""
<style>
.main { background-color: #0f172a; color: #e2e8f0; }
h1, h2, h3 { color: #f1f5f9; }
.stButton>button { background-color: #3b82f6; color: white; border-radius: 8px; height: 50px; font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# 설정 상수
CONFIG = {
    "MAX_CANDIDATES": 10,
    "TREND_DAYS": 180,
    "PREDICTION_DAYS": 30,
    "MAX_NEWS": 20,
    "TIMEOUT": 30,
    "COLORS": ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
}


def load_api_keys():
    """API 키 로드"""
    try:
        keys = {
            "gemini": st.secrets["GEMINI_API_KEY"],
            "naver_id": st.secrets["NAVER_CLIENT_ID"],
            "naver_secret": st.secrets["NAVER_CLIENT_SECRET"],
        }
        try:
            keys["apify"] = st.secrets["APIFY_API_KEY"]
        except:
            keys["apify"] = None
        return keys
    except Exception as e:
        logger.error(f"API 키 로드 실패: {str(e)}")
        st.error("API 키 로드 실패. `.streamlit/secrets.toml` 확인")
        st.stop()


def validate_candidates(candidates):
    """후보자 이름 유효성 검사"""
    if not candidates:
        return False, "후보자를 최소 1명 입력하세요"
    if len(candidates) > CONFIG["MAX_CANDIDATES"]:
        return False, f"후보자는 최대 {CONFIG['MAX_CANDIDATES']}명까지 가능합니다"
    for c in candidates:
        if len(c) < 2:
            return False, f"'{c}'는 너무 짧습니다 (최소 2자)"
    return True, ""


def get_best_model(api_key):
    """최적의 Gemini 모델 선택"""
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if 'flash' in m.lower():
                return m
        if models:
            return models[0]
    except Exception as e:
        logger.warning(f"모델 목록 조회 실패: {str(e)}")
    return "models/gemini-1.5-flash"


def clean_json(text):
    """JSON 텍스트 정제"""
    if not text:
        return None
    try:
        text = re.sub(r'```json\s*|```\s*', '', text).strip()
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0).replace('\n', ' '))
        return json.loads(text)
    except Exception as e:
        logger.warning(f"JSON 파싱 실패: {str(e)}")
        return None


def safe_get_last_value(df, col_name):
    """안전한 마지막 값 추출"""
    if df.empty or col_name not in df.columns or len(df) == 0:
        return None
    try:
        return df[col_name].iloc[-1]
    except Exception as e:
        logger.warning(f"값 추출 실패 ({col_name}): {str(e)}")
        return None


@st.cache_data(ttl=3600)
def get_naver_trend(candidates, n_id, n_secret):
    """네이버 트렌드 데이터 수집"""
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
        logger.error(f"네이버 트렌드 수집 실패: {str(e)}")
        return pd.DataFrame()


def get_google_trend_apify(candidates, api_key, status_container=None):
    """Apify Google Trends Scraper"""
    if not api_key:
        return pd.DataFrame()
    
    try:
        run_url = "https://api.apify.com/v2/acts/emastra~google-trends-scraper/runs?token=" + api_key
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
            run_data = resp.json()
            run_id = run_data.get("data", {}).get("id")
            
            if run_id:
                for i in range(45):
                    time.sleep(2)
                    if status_container:
                        status_container.info(f"📊 구글 트렌드 수집 중... ({i*2}/90초)")
                    
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
                                    items = items_resp.json()
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
                                                    except Exception as e:
                                                        logger.warning(f"데이터 파싱 실패: {e}")
                                                        continue
                                                
                                                if dates and values:
                                                    all_data[name] = pd.Series(values, index=dates)
                                    
                                    if all_data:
                                        df = pd.DataFrame(all_data)
                                        df.index = pd.to_datetime(df.index)
                                        return df.sort_index()
                            break
                        elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                            break
    except Exception as e:
        logger.error(f"구글 트렌드 수집 실패: {str(e)}")
    
    return pd.DataFrame()


def get_news_trend(candidates):
    """뉴스 언급량 기반 트렌드"""
    ddgs = DDGS()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=CONFIG["TREND_DAYS"])
    all_counts = {}
    
    for name in candidates:
        try:
            time.sleep(random.uniform(0.5, 1.0))
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
                    except:
                        continue
                
                if date_counts:
                    all_counts[name] = date_counts
        except Exception as e:
            logger.error(f"뉴스 수집 실패: {str(e)}")
            continue
    
    global_max = max([max(c.values()) for c in all_counts.values() if c], default=1)
    trend_data = {}
    
    for name, date_counts in all_counts.items():
        normalized = {k: (v / global_max) * 100 for k, v in date_counts.items()}
        trend_data[name] = normalized
    
    if trend_data:
        all_dates = set()
        current = start_date.date()
        while current <= end_date.date():
            week_start = current - timedelta(days=current.weekday())
            all_dates.add(week_start)
            current += timedelta(days=7)
        
        for data in trend_data.values():
            all_dates.update(data.keys())
        
        if all_dates:
            df_data = {name: [data.get(d, 0) for d in sorted(all_dates)] for name, data in trend_data.items()}
            df = pd.DataFrame(df_data, index=sorted(all_dates))
            df.index = pd.to_datetime(df.index)
            return df
    
    return pd.DataFrame()


def get_all_trends(candidates, keys, status_container):
    """모든 트렌드 데이터 수집"""
    results = {"naver": pd.DataFrame(), "google": pd.DataFrame()}
    
    status_container.info("📊 네이버 트렌드 수집 중...")
    results["naver"] = get_naver_trend(candidates, keys["naver_id"], keys["naver_secret"])
    
    if keys.get("apify"):
        results["google"] = get_google_trend_apify(candidates, keys["apify"], status_container)
    
    if results["google"].empty:
        status_container.info("📊 뉴스 언급량 수집 중...")
        results["google"] = get_news_trend(candidates)
    
    return results


def predict_future(df, days=30):
    """미래 트렌드 예측"""
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
        x, y = np.arange(len(recent)), recent.values
        
        if len(x) > 1:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                predictions = p(np.arange(len(recent), len(recent) + days))
                min_value = max(recent.iloc[-1] * 0.5, 1)
                future_df[col] = np.clip(predictions, min_value, 100)
            except Exception as e:
                logger.warning(f"예측 실패: {str(e)}")
    
    if not future_df.empty:
        future_df.index = future_dates
    
    return future_df


def contains_name(text, name):
    """텍스트에 이름 포함 여부"""
    if not text or not name:
        return False
    return name in text or (len(name) >= 2 and name[1:] in text)


def collect_all_data(keyword, election_name):
    """후보자 관련 모든 데이터 수집"""
    ddgs = DDGS()
    collected = {
        "news": {"text": [], "links": []},
        "sns": {"text": [], "links": []},
        "community": {"text": [], "links": []},
        "wiki": {"text": [], "links": []},
        "youtube": {"text": [], "links": []}
    }
    
    # 뉴스
    try:
        time.sleep(random.uniform(0.5, 1.0))
        news = ddgs.news(f'"{keyword}" 2025 OR 2026', region="kr-kr", safesearch="off", max_results=10)
        if not news or len(news) < 3:
            news = ddgs.news(f'"{keyword}" {election_name}', region="kr-kr", safesearch="off", max_results=10)
        
        for r in (news or []):
            title, body, date_str = r.get('title', ''), r.get('body', ''), r.get('date', '')
            if contains_name(title, keyword) or contains_name(body, keyword):
                collected["news"]["text"].append(f"[{date_str}] {title}: {body[:200]}")
                collected["news"]["links"].append({
                    "title": title[:50],
                    "url": r.get('url', '#'),
                    "source": r.get('source', ''),
                    "date": date_str,
                    "body": body[:300]
                })
    except Exception as e:
        logger.error(f"뉴스 수집 실패: {str(e)}")
    
    # 위키
    try:
        time.sleep(random.uniform(0.5, 1.0))
        profile = ddgs.text(f'"{keyword}" 정당 소속 현재 2025', region="kr-kr", safesearch="off", max_results=5)
        for r in (profile or []):
            title, body = r.get('title', ''), r.get('body', '')
            if contains_name(title, keyword) or contains_name(body, keyword):
                collected["wiki"]["text"].insert(0, f"{title}: {body[:250]}")
                collected["wiki"]["links"].insert(0, {"title": title[:50], "url": r.get('href', '#')})
    except Exception as e:
        logger.error(f"프로필 수집 실패: {str(e)}")
    
    try:
        time.sleep(random.uniform(0.5, 1.0))
        wiki = ddgs.text(f'"{keyword}" (site:namu.wiki OR site:ko.wikipedia.org)', region="kr-kr", safesearch="off", max_results=5)
        for r in (wiki or []):
            title, body = r.get('title', ''), r.get('body', '')
            if contains_name(title, keyword) or contains_name(body, keyword):
                collected["wiki"]["text"].append(f"{title}: {body[:300]}")
                collected["wiki"]["links"].append({"title": title[:50], "url": r.get('href', '#')})
    except Exception as e:
        logger.error(f"위키 수집 실패: {str(e)}")
    
    # SNS
    try:
        time.sleep(random.uniform(0.5, 1.0))
        sns = ddgs.text(f'"{keyword}" (site:blog.naver.com OR site:cafe.naver.com OR site:tistory.com)', region="kr-kr", safesearch="off", max_results=15)
        for r in (sns or []):
            title, body = r.get('title', ''), r.get('body', '')
            if contains_name(title, keyword) or contains_name(body, keyword):
                url = r.get('href', '')
                source = "네이버블로그" if 'blog.naver' in url else "네이버카페" if 'cafe.naver' in url else "티스토리" if 'tistory' in url else "SNS"
                collected["sns"]["text"].append(f"[{source}] {title}: {body[:150]}")
                collected["sns"]["links"].append({"title": title[:50], "url": url, "source": source})
    except Exception as e:
        logger.error(f"SNS 수집 실패: {str(e)}")
    
    # 커뮤니티
    try:
        time.sleep(random.uniform(0.5, 1.0))
        community = ddgs.text(f'"{keyword}" (site:dcinside.com OR site:clien.net)', region="kr-kr", safesearch="off", max_results=10)
        for r in (community or []):
            title, body = r.get('title', ''), r.get('body', '')
            if contains_name(title, keyword) or contains_name(body, keyword):
                url = r.get('href', '')
                source = "디시인사이드" if 'dcinside' in url else "클리앙" if 'clien' in url else "커뮤니티"
                collected["community"]["text"].append(f"[{source}] {title}: {body[:150]}")
                collected["community"]["links"].append({"title": title[:50], "url": url, "source": source})
    except Exception as e:
        logger.error(f"커뮤니티 수집 실패: {str(e)}")
    
    # 유튜브
    try:
        time.sleep(random.uniform(0.5, 1.0))
        videos = ddgs.videos(f'"{keyword}" {election_name}', region="kr-kr", safesearch="off", max_results=5)
        for r in (videos or []):
            title = r.get('title', '')
            if contains_name(title, keyword):
                collected["youtube"]["text"].append(title)
                collected["youtube"]["links"].append({"title": title[:50], "url": r.get('content', '#')})
    except Exception as e:
        logger.error(f"유튜브 수집 실패: {str(e)}")
    
    return collected


def analyze_candidate(model, name, collected_data, trend_info):
    """후보자 AI 분석"""
    news_cnt = len(collected_data.get("news", {}).get("links", []))
    sns_cnt = len(collected_data.get("sns", {}).get("links", []))
    community_cnt = len(collected_data.get("community", {}).get("links", []))
    wiki_cnt = len(collected_data.get("wiki", {}).get("links", []))
    
    def build_career_timeline(news_links):
        """뉴스 발행일 기반 경력 타임라인"""
        timeline = []
        
        for link in news_links:
            date_str = link.get('date', '')
            if not date_str:
                continue
            
            try:
                article_date = pd.to_datetime(date_str)
                content = f"{link.get('title', '')} {link.get('body', '')}"
                
                if article_date < pd.Timestamp('2013-02-25'):
                    government = "이명박 정부"
                elif article_date < pd.Timestamp('2017-05-10'):
                    government = "박근혜 정부"
                elif article_date < pd.Timestamp('2022-05-10'):
                    government = "문재인 정부"
                elif article_date < pd.Timestamp('2025-05-10'):
                    government = "윤석열 정부"
                else:
                    government = "현 정부"
                
                careers = []
                
                if "청와대" in content or "대통령실" in content:
                    if "청년위원장" in content:
                        careers.append(f"{government} 청년위원장")
                    elif "수석" in content:
                        careers.append(f"{government} 수석")
                    else:
                        careers.append(f"{government} 청와대" if government in ["이명박 정부", "박근혜 정부", "문재인 정부"] else f"{government} 대통령실")
                
                if "국회의원" in content:
                    for term in ["22대", "21대", "20대", "19대", "18대"]:
                        if term in content:
                            careers.append(f"{term} 국회의원")
                            break
                
                if "도지사" in content and "후보" not in content:
                    careers.append("광역단체장")
                if "장관" in content:
                    careers.append("장관")
                
                for career in careers:
                    timeline.append({"date": article_date, "career": career})
            except:
                continue
        
        timeline.sort(key=lambda x: x['date'], reverse=True)
        seen = set()
        unique = []
        for item in timeline:
            if item['career'] not in seen:
                seen.add(item['career'])
                unique.append(item['career'])
        
        return unique
    
    career_timeline = build_career_timeline(collected_data.get("news", {}).get("links", []))
    
    all_text = (
        collected_data.get("news", {}).get("text", [])[:5] + 
        collected_data.get("wiki", {}).get("text", [])[:5] +
        collected_data.get("sns", {}).get("text", [])[:3] + 
        collected_data.get("community", {}).get("text", [])[:2]
    )
    raw_text = "\n".join(all_text)
    
    party_patterns = {
        "더불어민주당": ["더불어민주당", "민주당 소속", "민주당 공천"],
        "국민의힘": ["국민의힘", "국민의힘 소속", "국민의힘 공천"],
        "개혁신당": ["개혁신당"],
        "무소속": ["무소속"]
    }
    
    detected_party = "정보 부족"
    max_matches = 0
    
    for party, patterns in party_patterns.items():
        match_count = sum(raw_text.count(p) for p in patterns)
        if match_count > max_matches:
            max_matches = match_count
            detected_party = party
    
    poll_est = 50
    try:
        if trend_info != "데이터 없음":
            match = re.search(r'현재\s*([\d.]+)', trend_info)
            if match:
                poll_est = min(int(float(match.group(1)) * 8), 100)
    except:
        pass
    
    if not raw_text or len(raw_text) < 30:
        return {
            "name": name,
            "party": "데이터 부족",
            "current_role": "정보 부족",
            "past_career": ", ".join(career_timeline[:3]) if career_timeline else "정보 부족",
            "poll_est": poll_est,
            "analysis": "수집된 데이터가 부족합니다.",
            "sns_sentiment": "분석 불가",
            "keywords": [name]
        }
    
    career_info = ", ".join(career_timeline[:5]) if career_timeline else "경력 정보 없음"
    
    prompt = f'''다음은 {name} 후보 정보입니다.

트렌드: {trend_info}
경력: {career_info}

데이터:
{raw_text[:2000]}

JSON 형식으로 응답하세요:
{{"name":"{name}","party":"정당명","current_role":"현재 직함","past_career":"주요 경력","poll_est":{poll_est},"analysis":"분석","sns_sentiment":"SNS 여론","keywords":["키워드"]}}'''

    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            result = clean_json(resp.text)
            if result and 'name' in result:
                if isinstance(result.get('poll_est'), str):
                    nums = re.findall(r'\d+', str(result['poll_est']))
                    result['poll_est'] = int(nums[0]) if nums else poll_est
                
                ai_party = result.get('party', '')
                if not ai_party or ai_party in ['정보 없음', '']:
                    result['party'] = detected_party
                
                if not result.get('past_career'):
                    result['past_career'] = ", ".join(career_timeline[:3]) if career_timeline else "정보 부족"
                
                return result
        except Exception as e:
            logger.warning(f"AI 실패 ({attempt+1}/3): {str(e)}")
            time.sleep(3)
    
    return {
        "name": name,
        "party": detected_party,
        "current_role": "충북지사 후보",
        "past_career": ", ".join(career_timeline[:3]) if career_timeline else "정보 부족",
        "poll_est": poll_est,
        "analysis": f"{name} 후보: 뉴스 {news_cnt}건 분석",
        "sns_sentiment": f"SNS/커뮤니티 {sns_cnt + community_cnt}건",
        "keywords": [name, detected_party] if detected_party != "정보 부족" else [name]
    }


def create_trend_chart(trends, pred_trends, candidates, google_source):
    """트렌드 차트 생성"""
    colors = CONFIG["COLORS"]
    naver_df = trends.get("naver", pd.DataFrame())
    google_df = trends.get("google", pd.DataFrame())
    pred_naver = pred_trends.get("naver", pd.DataFrame())
    pred_google = pred_trends.get("google", pd.DataFrame())
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("🟢 네이버 트렌드", f"🔵 {google_source}"), horizontal_spacing=0.08)
    
    if not naver_df.empty:
        for idx, col in enumerate(candidates):
            if col in naver_df.columns:
                color = colors[idx % len(colors)]
                fig.add_trace(go.Scatter(x=naver_df.index, y=naver_df[col], mode='lines', name=col,
                              line=dict(color=color, width=2), legendgroup=col), row=1, col=1)
                if not pred_naver.empty and col in pred_naver.columns:
                    pred_x = [naver_df.index[-1]] + list(pred_naver.index)
                    pred_y = [naver_df[col].iloc[-1]] + list(pred_naver[col])
                    fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines',
                                  line=dict(color=color, width=2, dash='dot'), legendgroup=col, showlegend=False), row=1, col=1)
    
    if not google_df.empty:
        for idx, col in enumerate(candidates):
            if col in google_df.columns:
                color = colors[idx % len(colors)]
                fig.add_trace(go.Scatter(x=google_df.index, y=google_df[col], mode='lines',
                              line=dict(color=color, width=2), legendgroup=col, showlegend=False), row=1, col=2)
                if not pred_google.empty and col in pred_google.columns:
                    pred_x = [google_df.index[-1]] + list(pred_google.index)
                    pred_y = [google_df[col].iloc[-1]] + list(pred_google[col])
                    fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines',
                                  line=dict(color=color, width=2, dash='dot'), legendgroup=col, showlegend=False), row=1, col=2)
    
    fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='#0f172a', plot_bgcolor='#1e293b',
                      font=dict(color='#e2e8f0'), legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                      margin=dict(l=50, r=50, t=50, b=80))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
    return fig


def get_party_color(party):
    """정당별 색상"""
    if not party:
        return "#4b5563"
    if "민주" in party:
        return "#1d4ed8"
    elif "국민의힘" in party:
        return "#b91c1c"
    elif "개혁" in party:
        return "#f97316"
    return "#4b5563"


def render_candidate_card(c, collected, trends, pred_trends, google_label):
    """후보자 카드 렌더링"""
    name = c.get('name', '미상')
    party = c.get('party', '정보 없음')
    party_color = get_party_color(party)
    
    naver_df = trends.get("naver", pd.DataFrame())
    google_df = trends.get("google", pd.DataFrame())
    pred_naver = pred_trends.get("naver", pd.DataFrame())
    pred_google = pred_trends.get("google", pd.DataFrame())
    
    badge = ""
    curr_val = safe_get_last_value(naver_df, name)
    fut_val = safe_get_last_value(pred_naver, name)
    
    if curr_val is not None and fut_val is not None:
        if fut_val > curr_val * 1.1:
            badge = "📈 상승예측"
        elif fut_val < curr_val * 0.9:
            badge = "📉 하락예측"
        else:
            badge = "➡️ 유지"
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            badge_html = f"<span style='background:#2563eb;color:white;padding:3px 8px;border-radius:10px;font-size:0.4em;margin-left:8px;'>{badge}</span>" if badge else ""
            st.markdown(f"### {name} <span style='background:{party_color};color:white;padding:4px 12px;border-radius:6px;font-size:0.5em;margin-left:10px;'>{party}</span>{badge_html}", unsafe_allow_html=True)
        with col2:
            st.metric("화제성", c.get('poll_est', 0))
        
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            val = safe_get_last_value(naver_df, name)
            st.metric("네이버 현재", f"{val:.1f}" if val is not None else "-")
        with tc2:
            val = safe_get_last_value(pred_naver, name)
            st.metric("네이버 예측", f"{val:.1f}" if val is not None else "-")
        with tc3:
            val = safe_get_last_value(google_df, name)
            st.metric(f"{google_label} 현재", f"{val:.1f}" if val is not None else "-")
        with tc4:
            val = safe_get_last_value(pred_google, name)
            st.metric(f"{google_label} 예측", f"{val:.1f}" if val is not None else "-")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔵 현재 직함**")
            st.info(c.get('current_role', '정보 없음'))
        with col2:
            st.markdown("**⚪ 주요 경력**")
            st.info(c.get('past_career', '정보 없음'))
        
        st.markdown("**📰 뉴스 분석**")
        st.write(c.get('analysis', ''))
        
        st.markdown("**💬 SNS 여론**")
        sns_sent = str(c.get('sns_sentiment', ''))
        if "긍정" in sns_sent:
            st.success(f"🟢 {sns_sent}")
        elif "부정" in sns_sent:
            st.error(f"🔴 {sns_sent}")
        else:
            st.warning(f"🟡 {sns_sent}")
        
        keywords = c.get('keywords', [])
        if keywords:
            st.markdown(" ".join([f"`#{k}`" for k in keywords[:5]]))
        
        st.markdown("**📚 데이터 출처**")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            f"📰 뉴스({len(collected.get('news',{}).get('links',[]))})",
            f"💬 SNS({len(collected.get('sns',{}).get('links',[]))})",
            f"👥 커뮤니티({len(collected.get('community',{}).get('links',[]))})",
            f"📖 위키({len(collected.get('wiki',{}).get('links',[]))})",
            f"📺 유튜브({len(collected.get('youtube',{}).get('links',[]))})"
        ])
        
        with tab1:
            for l in collected.get("news", {}).get("links", []):
                st.markdown(f"- [{l['title']}]({l['url']})")
        with tab2:
            for l in collected.get("sns", {}).get("links", []):
                st.markdown(f"- [{l['title']}]({l['url']})")
        with tab3:
            for l in collected.get("community", {}).get("links", []):
                st.markdown(f"- [{l['title']}]({l['url']})")
        with tab4:
            for l in collected.get("wiki", {}).get("links", []):
                st.markdown(f"- [{l['title']}]({l['url']})")
        with tab5:
            for l in collected.get("youtube", {}).get("links", []):
                st.markdown(f"- [{l['title']}]({l['url']})")
        
        st.divider()


def main():
    """메인 애플리케이션"""
    st.title("🏛️ 선거 전략 인사이트: 예측과 전망")
    st.caption("네이버 + 구글(Apify) 트렌드 종합 분석")
    
    keys = load_api_keys()
    
    with st.sidebar:
        st.success("✅ 시스템 정상 가동")
        
        if keys.get("apify"):
            st.info("🔵 Apify 연결됨")
        else:
            st.warning("⚠️ Apify 미설정")
        
        st.divider()
        
        election = st.text_input("분석 대상 선거", value="2026년 충청북도지사 선거")
        st.markdown("**후보자 목록**")
        cands_txt = st.text_area("", value="신용한\n노영민\n송기섭", height=150, label_visibility="collapsed")
        
        cands = [c.strip() for c in cands_txt.split('\n') if c.strip()]
        st.caption(f"등록: {len(cands)}명")
        
        is_valid, msg = validate_candidates(cands)
        if not is_valid:
            st.error(msg)
        
        start = st.button("🚀 종합 분석 실행", type="primary", use_container_width=True, disabled=not is_valid)

    if start:
        model = genai.GenerativeModel(get_best_model(keys["gemini"]))
        status = st.empty()
        progress = st.progress(0)
        
        try:
            trends = get_all_trends(cands, keys, status)
            progress.progress(0.25)
            
            google_source = "구글 트렌드" if keys.get("apify") and not trends["google"].empty else "뉴스 언급량"
            google_label = "구글" if "구글" in google_source else "뉴스"
            
            status.info("🔮 미래 예측 계산 중...")
            pred_trends = {
                "naver": predict_future(trends["naver"], days=CONFIG["PREDICTION_DAYS"]),
                "google": predict_future(trends["google"], days=CONFIG["PREDICTION_DAYS"])
            }
            progress.progress(0.3)
            
            results, all_collected = [], {}
            for i, name in enumerate(cands):
                status.info(f"⚡ [{i+1}/{len(cands)}] {name} 분석 중...")
                collected = collect_all_data(name, election)
                all_collected[name] = collected
                
                trend_info = "데이터 없음"
                curr_val = safe_get_last_value(trends.get("naver", pd.DataFrame()), name)
                fut_val = safe_get_last_value(pred_trends.get("naver", pd.DataFrame()), name)
                
                if curr_val is not None:
                    fut_val = fut_val if fut_val is not None else curr_val
                    trend_info = f"현재 {curr_val:.1f}, 예측 {fut_val:.1f}"
                
                results.append(analyze_candidate(model, name, collected, trend_info))
                progress.progress(0.3 + (0.65 * (i + 1) / len(cands)))
                time.sleep(2)
            
            status.empty()
            progress.empty()
            st.success("✅ 분석 완료!")
            
            st.subheader("📈 트렌드 예측 시뮬레이션")
            st.plotly_chart(create_trend_chart(trends, pred_trends, cands, google_source), use_container_width=True)
            
            st.markdown("**📊 트렌드 수치 요약**")
            naver_df = trends.get("naver", pd.DataFrame())
            google_df = trends.get("google", pd.DataFrame())
            pred_naver = pred_trends.get("naver", pd.DataFrame())
            pred_google = pred_trends.get("google", pd.DataFrame())
            
            summary = []
            for name in cands:
                row = {"후보": name}
                row["네이버 현재"] = f"{safe_get_last_value(naver_df, name):.1f}" if safe_get_last_value(naver_df, name) is not None else "-"
                row["네이버 예측"] = f"{safe_get_last_value(pred_naver, name):.1f}" if safe_get_last_value(pred_naver, name) is not None else "-"
                row[f"{google_label} 현재"] = f"{safe_get_last_value(google_df, name):.1f}" if safe_get_last_value(google_df, name) is not None else "-"
                row[f"{google_label} 예측"] = f"{safe_get_last_value(pred_google, name):.1f}" if safe_get_last_value(pred_google, name) is not None else "-"
                summary.append(row)
            
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
            st.divider()
            
            st.subheader("📋 후보자별 심층 리포트")
            for c in results:
                render_candidate_card(c, all_collected.get(c['name'], {}), trends, pred_trends, google_label)
        
        except Exception as e:
            status.empty()
            progress.empty()
            st.error(f"❌ 오류 발생: {str(e)}")
            logger.error(f"메인 오류: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
