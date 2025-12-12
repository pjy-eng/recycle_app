import streamlit as st
import time
from PIL import Image
from datetime import datetime
import plotly.express as px
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ==================================================
# 页面配置
# ==================================================
st.set_page_config("SmartRecycle", "♻️", layout="wide")

# ==================================================
# Session State
# ==================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_points" not in st.session_state:
    st.session_state.total_points = 0

# ==================================================
# 多语言
# ==================================================
TRANS = {
    "zh": {
        "home": "首页",
        "scan_title": "垃圾识别，从一张照片开始",
        "scan_sub": "拍照 / 上传 → AI识别 → 分类 → 获得积分",
        "upload": "📂 上传图片（支持多张）",
        "camera": "📷 拍照（建议光线充足）",
        "start": "开始识别",
        "low": "识别置信度较低，仅供参考",
        "dashboard": "数据看板",
        "points": "积分系统",
        "history": "记录",
        "level": "当前等级",
    },
    "en": {
        "home": "Home",
        "scan_title": "Recycle smarter with one photo",
        "scan_sub": "Upload / Camera → AI → Sort → Earn points",
        "upload": "📂 Upload images (multiple)",
        "camera": "📷 Camera (good lighting recommended)",
        "start": "Start Scan",
        "low": "Low confidence, for reference only",
        "dashboard": "Dashboard",
        "points": "Points",
        "history": "History",
        "level": "Level",
    },
    "kr": {
        "home": "홈",
        "scan_title": "사진 한 장으로 쓰레기 분류",
        "scan_sub": "업로드 / 촬영 → AI 인식 → 분리배출 → 포인트",
        "upload": "📂 이미지 업로드 (여러 장)",
        "camera": "📷 카메라 촬영 (밝은 환경 권장)",
        "start": "인식 시작",
        "low": "신뢰도가 낮아 참고용입니다",
        "dashboard": "데이터",
        "points": "포인트",
        "history": "기록",
        "level": "레벨",
    }
}

# ==================================================
# 模型
# ==================================================
@st.cache_resource
def load_model():
    model_id = "yangy50/garbage-classification"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model

processor, model = load_model()

# ==================================================
# label → UI
# ==================================================
LABEL_UI = {
    "plastic": ("Plastic / 塑料", "🥤", "#10b981", 10),
    "paper": ("Paper / 纸类", "📰", "#f59e0b", 5),
    "metal": ("Metal / 金属", "🥫", "#3b82f6", 15),
    "glass": ("Glass / 玻璃", "🍾", "#a855f7", 10),
    "cardboard": ("Cardboard / 纸板", "📦", "#f59e0b", 5),
    "trash": ("Trash / 一般垃圾", "🗑️", "#64748b", 1),
    "unknown": ("Unknown / 未知", "❓", "#94a3b8", 0),
}

# ==================================================
# 分类（batch）
# ==================================================
def classify_batch(images):
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    scores, ids = torch.max(probs, dim=-1)

    results = []
    for s, i in zip(scores, ids):
        key = model.config.id2label[i.item()]
        if s.item() < 0.35:
            key = "unknown"
        results.append((key, s.item()))
    return results

# ==================================================
# Sidebar（语言 + 积分）
# ==================================================
with st.sidebar:
    lang = st.selectbox(
        "Language / 언어",
        ["zh", "en", "kr"],
        format_func=lambda x: {"zh": "🇨🇳 中文", "en": "🇺🇸 English", "kr": "🇰🇷 한국어"}[x]
    )
    t = TRANS[lang]
    st.metric("⭐ Points", st.session_state.total_points)

# ==================================================
# 顶部导航
# ==================================================
tab_home, tab_dashboard, tab_points, tab_history = st.tabs(
    [t["home"], t["dashboard"], t["points"], t["history"]]
)

# ==================================================
# 首页 = 识别
# ==================================================
with tab_home:
    st.markdown(f"## ♻️ {t['scan_title']}")
    st.caption(t["scan_sub"])

    col1, col2 = st.columns(2)
    images = []

    with col1:
        files = st.file_uploader(
            t["upload"], type=["jpg", "png", "jpeg"], accept_multiple_files=True
        )
        if files:
            for f in files:
                images.append(Image.open(f).convert("RGB"))

    with col2:
        cam = st.camera_input(t["camera"])
        if cam:
            img = Image.open(cam).convert("RGB").resize((384, 384))
            images.append(img)

    if images and st.button(t["start"], use_container_width=True):
        with st.spinner("AI analyzing..."):
            time.sleep(1)

        results = classify_batch(images)

        for img, (key, score) in zip(images, results):
            name, icon, color, pts = LABEL_UI[key]
            st.session_state.total_points += pts
            st.session_state.history.insert(0, {
                "label": name,
                "points": pts,
                "time": datetime.now().strftime("%H:%M")
            })

            st.markdown(f"""
            <div style="margin:20px 0;padding:20px;border-radius:16px;
            background:linear-gradient(135deg,{color}33,#111);text-align:center;">
            <div style="font-size:4rem">{icon}</div>
            <h3>{name}</h3>
            <b>+{pts} pts</b>
            </div>
            """, unsafe_allow_html=True)

            if score < 0.5:
                st.caption("⚠️ " + t["low"])

        st.balloons()

# ==================================================
# 数据看板
# ==================================================
with tab_dashboard:
    if st.session_state.history:
        counter = {}
        for h in st.session_state.history:
            counter[h["label"]] = counter.get(h["label"], 0) + 1

        fig = px.pie(names=counter.keys(), values=counter.values(), hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

# ==================================================
# 积分系统
# ==================================================
with tab_points:
    level = st.session_state.total_points // 100 + 1
    st.metric(t["level"], level)
    st.progress((st.session_state.total_points % 100) / 100)

# ==================================================
# 记录
# ==================================================
with tab_history:
    for h in st.session_state.history:
        st.markdown(f"- **{h['label']}** ｜ +{h['points']} ｜ {h['time']}")
