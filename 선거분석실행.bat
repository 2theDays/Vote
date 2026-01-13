@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ========================================
echo   선거 전략 인사이트 v2.0
echo   네이버 + 구글 트렌드 종합 분석
echo ========================================
echo.
echo 시스템을 시작합니다...
echo (이 창을 닫으면 프로그램이 종료됩니다)
echo.
echo 브라우저에서 자동으로 열립니다.
echo 열리지 않으면 http://localhost:8501 접속
echo.

streamlit run app.py --server.port 8501

pause
