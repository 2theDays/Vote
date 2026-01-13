# 선거 전략 인사이트: 예측과 전망

네이버 + 구글(Apify) 트렌드 종합 분석 시스템

## 기능

- 📊 트렌드 분석: 네이버 데이터랩, 구글 트렌드, 뉴스 언급량
- 🤖 AI 분석: Gemini 기반 후보자 분석
- 📈 시각화: 트렌드 차트, 레이더 차트, 비교 분석
- 💾 데이터 내보내기: JSON 형식으로 결과 저장

## 설치 및 실행

### 로컬 실행

1. Python 3.11 이상 설치
2. 패키지 설치:
```bash
pip install -r requirements.txt
```

3. API 키 설정:
   - `.streamlit/secrets.toml` 파일 생성
   - 예제 파일 참고: `.streamlit/secrets.toml.example`

4. 실행:
```bash
streamlit run app.py
```

또는 Windows에서:
```bash
선거분석실행.bat
```

## 배포

### Streamlit Cloud 배포 (권장)

Streamlit 앱은 Streamlit Cloud에서 가장 잘 작동합니다.

1. GitHub에 코드 푸시:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. [Streamlit Cloud](https://streamlit.io/cloud) 접속
3. "New app" 클릭
4. GitHub 저장소 선택
5. Main file path: `app.py`
6. Advanced settings에서 Secrets 추가:
   ```
   GEMINI_API_KEY=your-key
   NAVER_CLIENT_ID=your-id
   NAVER_CLIENT_SECRET=your-secret
   APIFY_API_KEY=your-key
   ```
7. "Deploy" 클릭

### Vercel 배포 (제한적)

⚠️ **중요**: Streamlit 앱은 Vercel에서 완전히 작동하지 않습니다. Streamlit은 지속적인 서버 연결이 필요하지만, Vercel은 서버리스 함수만 지원합니다.

**Streamlit Cloud 사용을 강력히 권장합니다.**

만약 Vercel을 사용해야 한다면:
1. Vercel CLI 설치:
```bash
npm i -g vercel
```

2. 배포:
```bash
vercel
```

3. 환경 변수 설정 (Vercel 대시보드):
   - `GEMINI_API_KEY`
   - `NAVER_CLIENT_ID`
   - `NAVER_CLIENT_SECRET`
   - `APIFY_API_KEY`

**참고**: Vercel 배포 시 Streamlit 앱이 정상 작동하지 않을 수 있습니다.

## 필요한 API 키

- **Gemini API**: Google AI Studio에서 발급
- **네이버 API**: 네이버 개발자 센터에서 발급
- **Apify API** (선택사항): Apify에서 발급

## 라이선스

MIT
