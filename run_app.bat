@echo off
title AeroGuard - Air Quality ML & RAG Copilot
echo =======================================================
echo AeroGuard: Air Quality Prediction & Grounded Copilot
echo MCA Combined Practical Assignment
echo Running exclusively from D: Drive
echo =======================================================
cd /d "D:\Gen AI"
set "PIP_CACHE_DIR=D:\Gen AI\.cache\pip"
set "TEMP=D:\Gen AI\.temp"
set "TMP=D:\Gen AI\.temp"

echo Activating Python virtual environment...
call "D:\Gen AI\venv\Scripts\activate.bat"

echo Launching Streamlit Dashboard...
streamlit run app.py
pause
