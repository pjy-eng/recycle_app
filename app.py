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
# Session
# ==================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_points" not in st.session_state:
    st.session_state.total_points = 0

# ==================================================
# 多语言
# ==================================================
LANG = {
    "zh": {
        "title": "垃圾识别，从一张照片开始",
        "sub": "拍照 / 上传 → AI识别 → 分类 → 获得积分",
        "upload": "📂 上传图片（支持多张）",
        "camera": "📷 拍照（建议光线充足）",
        "start": "开始识别",
        "low": "识别置信度较低，仅供参考",
        "data": "我的数据",
        "history": "记录"
    }
}
t = LANG["zh"]

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
    "plastic": ("塑料", "🥤", "#10b981", 10),
    "paper": ("纸类", "📰", "#f59e0b", 5),
    "metal": ("金属", "🥫", "#3b82f6", 15),
    "glass": ("玻璃", "🍾", "#a855f7", 10),
    "cardboard": ("纸板", "📦", "#f59e0b", 5),
    "trash": ("一般垃圾", "🗑️", "#64748b", 1),
    "unknown": ("无法识别", "❓", "#94a3b8", 0),
}

# ==================================================
# 分类
# ==================================================
def classify_batch(images):
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    scores, ids = torch.max(probs, dim=-1)

    results = []
    for score, idx in zip(scores, ids):
        key = model.config.id2label[idx.item()]
        if score.item() < 0.35:
            key = "unknown"
        results.append((key, score.item()))
    return results

# ==================================================
# 首页 = 识别页
# ==================================================
st.markdown(f"## ♻️ {t['title']}")
st.caption(t["sub"])

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
        img = Image.open(cam).convert("RGB")
        img = img.resize((384, 384))  # 🔥 降低拍照噪声
        images.append(img)

if images:
    if st.button(t["start"], use_container_width=True):
        with st.spinner("AI 分析中…"):
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
            <div style="
            margin:20px 0;
            padding:20px;
            border-radius:16px;
            background:linear-gradient(135deg,{color}33,#111);
            text-align:center;">
                <div style="font-size:4rem">{icon}</div>
                <h3>{name}</h3>
                <b>+{pts} pts</b>
            </div>
            """, unsafe_allow_html=True)

            if score < 0.5:
                st.caption("⚠️ " + t["low"])

        st.balloons()

# ==================================================
# 数据
# ==================================================
st.divider()
st.subheader("📊 我的数据")

if st.session_state.history:
    counter = {}
    for h in st.session_state.history:
        counter[h["label"]] = counter.get(h["label"], 0) + 1

    fig = px.pie(names=counter.keys(), values=counter.values(), hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("暂无数据")
