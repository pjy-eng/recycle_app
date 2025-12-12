import streamlit as st
import time
from PIL import Image
from datetime import datetime
import plotly.express as px
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ==================================================
# 1. 页面配置
# ==================================================
st.set_page_config(
    page_title="SmartRecycle",
    page_icon="♻️",
    layout="wide"
)

# ==================================================
# 2. Session State
# ==================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_points" not in st.session_state:
    st.session_state.total_points = 0
if "last_res" not in st.session_state:
    st.session_state.last_res = None

# ==================================================
# 3. 多语言
# ==================================================
TRANS = {
    "zh": {
        "home": "首页",
        "scan": "开始识别",
        "data": "我的数据",
        "history": "记录",
        "hero_title": "垃圾识别，从一张照片开始",
        "hero_sub": "拍照 → AI识别 → 正确分类 → 获得积分",
        "cta": "👉 立即开始识别",
        "upload": "上传或拍摄垃圾照片",
        "start": "AI 识别",
        "analyzing": "AI 正在分析中…",
        "result": "AI 建议分类",
        "points": "本次获得积分",
        "low_conf": "识别置信度较低，仅供参考"
    },
    "en": {
        "home": "Home",
        "scan": "Scan",
        "data": "My Data",
        "history": "History",
        "hero_title": "Recycle smarter with one photo",
        "hero_sub": "Photo → AI → Learn → Earn points",
        "cta": "👉 Start Scanning",
        "upload": "Upload or take a photo",
        "start": "AI Scan",
        "analyzing": "AI is analyzing…",
        "result": "AI Suggested Category",
        "points": "Points Earned",
        "low_conf": "Low confidence, for reference only"
    },
    "kr": {
        "home": "홈",
        "scan": "AI 인식",
        "data": "내 데이터",
        "history": "기록",
        "hero_title": "사진 한 장으로 쓰레기 분류",
        "hero_sub": "촬영 → AI 인식 → 분리배출 → 포인트 획득",
        "cta": "👉 스캔 시작",
        "upload": "쓰레기 사진 업로드",
        "start": "AI 인식",
        "analyzing": "AI 분석 중…",
        "result": "AI 분류 제안",
        "points": "획득 포인트",
        "low_conf": "신뢰도가 낮아 참고용입니다"
    }
}

# ==================================================
# 4. 加载垃圾分类模型
# ==================================================
@st.cache_resource
def load_model():
    MODEL = "yangy50/garbage-classification"
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForImageClassification.from_pretrained(MODEL)
    model.eval()
    return processor, model

processor, model = load_model()

# ==================================================
# 5. id2label → UI 映射
# ==================================================
WASTE_UI = {
    "plastic": ("🥤", "#10b981", 10),
    "paper": ("📰", "#f59e0b", 5),
    "metal": ("🥫", "#3b82f6", 15),
    "glass": ("🍾", "#a855f7", 10),
    "cardboard": ("📦", "#f59e0b", 5),
    "trash": ("🗑️", "#64748b", 1),
    "unknown": ("❓", "#94a3b8", 0)
}

# ==================================================
# 6. 分类函数
# ==================================================
def classify(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    score, pred_id = torch.max(probs, dim=-1)
    score = score.item()
    key = model.config.id2label[pred_id.item()]

    if score < 0.35:
        key = "unknown"

    icon, color, points = WASTE_UI[key]
    return key, icon, color, points, score

# ==================================================
# 7. 侧边栏（仅辅助）
# ==================================================
with st.sidebar:
    lang = st.selectbox(
        "Language / 언어",
        ["zh", "en", "kr"],
        format_func=lambda x: {"zh": "🇨🇳 中文", "en": "🇺🇸 English", "kr": "🇰🇷 한국어"}[x]
    )
    t = TRANS[lang]

# ==================================================
# 8. 顶部导航
# ==================================================
tab_home, tab_scan, tab_data, tab_history = st.tabs(
    [t["home"], t["scan"], t["data"], t["history"]]
)

# ==================================================
# 9. 右上角积分悬浮窗
# ==================================================
st.markdown(f"""
<div style="
position: fixed;
top: 15px;
right: 25px;
background: #10b981;
color: white;
padding: 10px 18px;
border-radius: 999px;
font-weight: bold;
z-index: 1000;
">
⭐ {st.session_state.total_points} pts
</div>
""", unsafe_allow_html=True)

# ==================================================
# 10. 首页（强主线）
# ==================================================
with tab_home:
    st.markdown(f"""
    <h1 style="font-size:3rem;">♻️ {t['hero_title']}</h1>
    <p style="font-size:1.4rem; opacity:0.8;">{t['hero_sub']}</p>
    """, unsafe_allow_html=True)

    if st.button(t["cta"], type="primary"):
        st.session_state.active_tab = "scan"

# ==================================================
# 11. 识别页（强反馈）
# ==================================================
with tab_scan:
    st.markdown(f"## 📸 {t['upload']}")

    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, width=320)

        if st.button(t["start"], use_container_width=True):
            with st.spinner(t["analyzing"]):
                time.sleep(1)

            key, icon, color, points, score = classify(img)
            st.session_state.total_points += points

            st.session_state.last_res = {
                "key": key,
                "icon": icon,
                "color": color,
                "points": points,
                "score": score
            }

            st.session_state.history.insert(0, {
                "key": key,
                "points": points,
                "time": datetime.now().strftime("%H:%M")
            })

            st.balloons()

    if st.session_state.last_res:
        r = st.session_state.last_res
        st.divider()

        st.markdown(f"""
        <div style="
        border-radius: 20px;
        padding: 30px;
        background: linear-gradient(135deg, {r['color']}33, #111);
        text-align: center;
        ">
            <div style="font-size:5rem;">{r['icon']}</div>
            <h2>{r['key'].upper()}</h2>
            <h3>+{r['points']} pts</h3>
        </div>
        """, unsafe_allow_html=True)

        if r["score"] < 0.5:
            st.caption("⚠️ " + t["low_conf"])

# ==================================================
# 12. 数据页
# ==================================================
with tab_data:
    if st.session_state.history:
        counter = {}
        for h in st.session_state.history:
            counter[h["key"]] = counter.get(h["key"], 0) + 1

        fig = px.pie(names=counter.keys(), values=counter.values(), hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

# ==================================================
# 13. 历史页
# ==================================================
with tab_history:
    for h in st.session_state.history:
        st.markdown(f"- **{h['key']}** ｜ +{h['points']} ｜ {h['time']}")
