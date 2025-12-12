import streamlit as st
import time
from PIL import Image
from datetime import datetime
import plotly.express as px

# =============================
# 1. 页面配置
# =============================
st.set_page_config(
    page_title="SmartRecycle",
    page_icon="♻️",
    layout="wide"
)

# =============================
# 2. Session State
# =============================
if "points" not in st.session_state:
    st.session_state.points = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =============================
# 3. 多语言配置（中 / 英 / 韩）
# =============================
TRANS = {
    "zh": {
        "home_title": "垃圾识别，从一张照片开始",
        "home_sub": "拍照 → AI识别 → 学会正确分类 → 获得积分",
        "start": "开始识别",
        "scan_title": "上传或拍摄垃圾照片",
        "scan_btn": "🔍 AI 识别",
        "analyzing": "AI 正在分析中…",
        "result": "识别结果",
        "points": "获得积分",
        "preview": "图片预览",
        "data_title": "我的数据",
        "history_title": "识别记录",
        "most_type": "我最常识别的垃圾",
    },
    "en": {
        "home_title": "Recycle smarter with one photo",
        "home_sub": "Photo → AI → Learn → Earn points",
        "start": "Start Scanning",
        "scan_title": "Upload or take a photo",
        "scan_btn": "🔍 AI Scan",
        "analyzing": "AI is analyzing…",
        "result": "Result",
        "points": "Points Earned",
        "preview": "Image Preview",
        "data_title": "My Statistics",
        "history_title": "History",
        "most_type": "Most scanned waste type",
    },
    "kr": {
        "home_title": "사진 한 장으로 쓰레기 분류",
        "home_sub": "촬영 → AI 인식 → 올바른 분리배출 → 포인트 획득",
        "start": "스캔 시작",
        "scan_title": "쓰레기 사진을 업로드하세요",
        "scan_btn": "🔍 AI 인식",
        "analyzing": "AI 분석 중…",
        "result": "인식 결과",
        "points": "획득 포인트",
        "preview": "이미지 미리보기",
        "data_title": "나의 데이터",
        "history_title": "기록",
        "most_type": "가장 많이 인식한 쓰레기",
    }
}

# =============================
# 4. 侧边栏（语言 + 积分）
# =============================
with st.sidebar:
    lang = st.selectbox(
        "Language / 언어",
        options=["zh", "en", "kr"],
        format_func=lambda x: {
            "zh": "🇨🇳 中文",
            "en": "🇺🇸 English",
            "kr": "🇰🇷 한국어"
        }[x]
    )
    t = TRANS[lang]

    st.markdown("---")
    st.metric("⭐ Points", st.session_state.points)

# =============================
# 5. 顶部导航
# =============================
tab_home, tab_scan, tab_data, tab_history = st.tabs(
    ["🏠 Home", "📸 Scan", "📊 Data", "📜 History"]
)

# =============================
# 6. 首页
# =============================
with tab_home:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            f"""
            <h1 style="font-size:3rem;">♻️ {t['home_title']}</h1>
            <p style="font-size:1.3rem; opacity:0.8;">{t['home_sub']}</p>
            """,
            unsafe_allow_html=True
        )

        st.info("👉 " + t["scan_title"])
        st.button(t["start"], type="primary", use_container_width=True)

    with col2:
        st.markdown("### 🌱")
        st.write("Make recycling easier and smarter.")

# =============================
# 7. AI 识别页（核心）
# =============================
with tab_scan:
    st.markdown(f"## 📸 {t['scan_title']}")

    img_file = st.file_uploader(
        "",
        type=["jpg", "png", "jpeg"],
        help=t["scan_title"]
    )

    if img_file:
        img = Image.open(img_file)
        st.image(img, width=320, caption=t["preview"])

        if st.button(t["scan_btn"], use_container_width=True):
            with st.spinner(t["analyzing"]):
                time.sleep(1)

            # ====== 模拟识别结果（你可以换成真实模型） ======
            label = {
                "zh": "塑料瓶",
                "en": "Plastic Bottle",
                "kr": "플라스틱 병"
            }[lang]

            advice = {
                "zh": "请清洗后放入塑料回收桶",
                "en": "Please clean and put it into the plastic recycling bin",
                "kr": "세척 후 플라스틱 수거함에 버려주세요"
            }[lang]

            points = 10

            # 保存结果
            st.session_state.last_result = {
                "label": label,
                "advice": advice,
                "points": points
            }

            st.session_state.points += points
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "label": label,
                "points": points
            })

            st.balloons()
            st.toast(f"🎉 +{points}")

    # ====== 结果展示（关键闭环） ======
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        st.markdown(f"### ✅ {t['result']}")
        st.success(res["label"])
        st.info(res["advice"])
        st.metric(t["points"], f"+{res['points']}")

# =============================
# 8. 数据页
# =============================
with tab_data:
    st.markdown(f"## 📊 {t['data_title']}")

    if st.session_state.history:
        counter = {}
        for h in st.session_state.history:
            counter[h["label"]] = counter.get(h["label"], 0) + 1

        fig = px.pie(
            names=counter.keys(),
            values=counter.values(),
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        most = max(counter, key=counter.get)
        st.success(f"{t['most_type']}：{most}")
    else:
        st.info("No data yet.")

# =============================
# 9. 历史记录
# =============================
with tab_history:
    st.markdown(f"## 📜 {t['history_title']}")

    if not st.session_state.history:
        st.info("No history.")
    else:
        for h in reversed(st.session_state.history):
            st.markdown(
                f"- **{h['label']}** ｜ +{h['points']} ｜ {h['time']}"
            )
