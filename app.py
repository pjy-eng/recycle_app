import streamlit as st
import time
from PIL import Image
from datetime import datetime
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from streamlit_lottie import st_lottie
import requests
import plotly.express as px

# =====================
# 1. 页面配置
# =====================
st.set_page_config(
    page_title="SmartRecycle",
    page_icon="♻️",
    layout="wide"
)

# =====================
# 2. Session State
# =====================
if "points" not in st.session_state:
    st.session_state.points = 0
if "history" not in st.session_state:
    st.session_state.history = []

# =====================
# 3. 多语言
# =====================
TRANS = {
    "zh": {
        "home_title": "垃圾识别，从一张照片开始",
        "home_sub": "拍照 → AI识别 → 学会正确分类 → 获得积分",
        "start": "开始识别",
        "scan_title": "上传或拍摄垃圾照片",
        "analyzing": "AI 正在分析中…",
        "points": "我的积分",
        "data_title": "我的使用数据",
        "history_title": "识别记录",
        "most_type": "我最常识别的垃圾",
    },
    "en": {
        "home_title": "Recycle smarter with one photo",
        "home_sub": "Photo → AI → Learn → Earn points",
        "start": "Start Scanning",
        "scan_title": "Upload or take a photo",
        "analyzing": "AI is analyzing…",
        "points": "My Points",
        "data_title": "My Statistics",
        "history_title": "History",
        "most_type": "Most scanned waste type",
    }
}

# =====================
# 4. 侧边栏（只保留必要内容）
# =====================
with st.sidebar:
    lang = st.selectbox("Language", ["zh", "en"])
    t = TRANS[lang]

    st.markdown("---")
    st.metric(t["points"], st.session_state.points)

# =====================
# 5. 顶部导航
# =====================
tab_home, tab_scan, tab_data, tab_history = st.tabs(
    ["🏠 Home", "📸 Scan", "📊 Data", "📜 History"]
)

# =====================
# 6. 首页
# =====================
with tab_home:
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown(f"""
        <h1 style="font-size:3rem;">♻️ {t["home_title"]}</h1>
        <p style="font-size:1.3rem; opacity:0.8;">{t["home_sub"]}</p>
        """, unsafe_allow_html=True)

        if st.button(t["start"], type="primary", use_container_width=True):
            st.experimental_set_query_params(tab="scan")

    with col2:
        lottie = requests.get("https://assets10.lottiefiles.com/packages/lf20_u4yrau.json").json()
        st_lottie(lottie, height=300)

# =====================
# 7. AI 识别页
# =====================
with tab_scan:
    st.markdown(f"## 📸 {t['scan_title']}")

    img_file = st.file_uploader("", type=["jpg","png","jpeg"])
    if img_file:
        img = Image.open(img_file)
        st.image(img, use_container_width=True)

        if st.button("🔍 AI 识别", use_container_width=True):
            with st.spinner(t["analyzing"]):
                time.sleep(1)

            # 模拟识别结果
            label = "Plastic Bottle"
            st.session_state.points += 10
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "label": label,
                "points": 10
            })

            st.balloons()
            st.toast("🎉 +10 Points!")

# =====================
# 8. 数据页
# =====================
with tab_data:
    st.markdown(f"## 📊 {t['data_title']}")

    if st.session_state.history:
        df = {}
        for h in st.session_state.history:
            df[h["label"]] = df.get(h["label"], 0) + 1

        fig = px.pie(
            names=df.keys(),
            values=df.values(),
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        most = max(df, key=df.get)
        st.success(f"{t['most_type']}：{most}")

# =====================
# 9. 历史记录
# =====================
with tab_history:
    st.markdown(f"## 📜 {t['history_title']}")

    for h in reversed(st.session_state.history):
        st.markdown(
            f"- **{h['label']}** ｜ +{h['points']} ｜ {h['time']}"
        )
