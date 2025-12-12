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
        "upload": "选择识别方式",
        "upload_tab": "📂 上传图片",
        "camera_tab": "📷 拍照",
        "start": "AI 识别",
        "analyzing": "AI 正在分析中…",
        "result": "识别结果",
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
        "upload": "Choose input method",
        "upload_tab": "📂 Upload",
        "camera_tab": "📷 Camera",
        "start": "AI Scan",
        "analyzing": "AI is analyzing…",
        "result": "Result",
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
        "upload": "입력 방식 선택",
        "upload_tab": "📂 이미지 업로드",
        "camera_tab": "📷 카메라 촬영",
        "start": "AI 인식",
        "analyzing": "AI 분석 중…",
        "result": "인식 결과",
        "points": "획득 포인트",
        "low_conf": "신뢰도가 낮아 참고용입니다"
    }
}

# ==================================================
# 4. 加载模型
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
# 5. label → UI 映射
# ==================================================
LABEL_UI = {
    "plastic": {
        "zh": ("塑料", "🥤", "#10b981", 10),
        "en": ("Plastic", "🥤", "#10b981", 10),
        "kr": ("플라스틱", "🥤", "#10b981", 10)
    },
    "paper": {
        "zh": ("纸类", "📰", "#f59e0b", 5),
        "en": ("Paper", "📰", "#f59e0b", 5),
        "kr": ("종이", "📰", "#f59e0b", 5)
    },
    "metal": {
        "zh": ("金属", "🥫", "#3b82f6", 15),
        "en": ("Metal", "🥫", "#3b82f6", 15),
        "kr": ("금속", "🥫", "#3b82f6", 15)
    },
    "glass": {
        "zh": ("玻璃", "🍾", "#a855f7", 10),
        "en": ("Glass", "🍾", "#a855f7", 10),
        "kr": ("유리", "🍾", "#a855f7", 10)
    },
    "cardboard": {
        "zh": ("纸板", "📦", "#f59e0b", 5),
        "en": ("Cardboard", "📦", "#f59e0b", 5),
        "kr": ("골판지", "📦", "#f59e0b", 5)
    },
    "trash": {
        "zh": ("一般垃圾", "🗑️", "#64748b", 1),
        "en": ("Trash", "🗑️", "#64748b", 1),
        "kr": ("일반 쓰레기", "🗑️", "#64748b", 1)
    },
    "unknown": {
        "zh": ("无法识别", "❓", "#94a3b8", 0),
        "en": ("Uncertain", "❓", "#94a3b8", 0),
        "kr": ("인식 불가", "❓", "#94a3b8", 0)
    }
}

# ==================================================
# 6. 分类函数
# ==================================================
def classify(image, lang):
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

    name, icon, color, points = LABEL_UI[key][lang]
    return name, icon, color, points, score, key

# ==================================================
# 7. 侧边栏
# ==================================================
with st.sidebar:
    lang = st.selectbox(
        "Language / 언어",
        ["zh", "en", "kr"],
        format_func=lambda x: {"zh": "🇨🇳 中文", "en": "🇺🇸 English", "kr": "🇰🇷 한국어"}[x]
    )
    st.metric("⭐ Points", st.session_state.total_points)
    t = TRANS[lang]

# ==================================================
# 8. 顶部导航
# ==================================================
tab_home, tab_scan, tab_data, tab_history = st.tabs(
    [t["home"], t["scan"], t["data"], t["history"]]
)

# ==================================================
# 9. 首页
# ==================================================
with tab_home:
    st.markdown(f"""
    <h1 style="font-size:3rem;">♻️ {t['hero_title']}</h1>
    <p style="font-size:1.4rem;">{t['hero_sub']}</p>
    """, unsafe_allow_html=True)

# ==================================================
# 10. 识别页（上传 + 拍照）
# ==================================================
with tab_scan:
    st.markdown(f"## 📸 {t['upload']}")

    up_tab, cam_tab = st.tabs([t["upload_tab"], t["camera_tab"]])

    img = None

    with up_tab:
        file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        if file:
            img = Image.open(file)
            st.image(img, width=320)

    with cam_tab:
        cam = st.camera_input("")
        if cam:
            img = Image.open(cam)
            st.image(img, width=320)

    if img:
        if st.button(t["start"], use_container_width=True):
            with st.spinner(t["analyzing"]):
                time.sleep(1)

            name, icon, color, points, score, key = classify(img, lang)

            st.session_state.total_points += points
            st.session_state.last_res = {
                "name": name,
                "icon": icon,
                "color": color,
                "points": points,
                "score": score
            }

            st.session_state.history.insert(0, {
                "label": name,
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
            <h2>{r['name']}</h2>
            <h3>+{r['points']} pts</h3>
        </div>
        """, unsafe_allow_html=True)

        if r["score"] < 0.5:
            st.caption("⚠️ " + t["low_conf"])

# ==================================================
# 11. 数据页
# ==================================================
with tab_data:
    if st.session_state.history:
        counter = {}
        for h in st.session_state.history:
            counter[h["label"]] = counter.get(h["label"], 0) + 1

        fig = px.pie(names=counter.keys(), values=counter.values(), hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

# ==================================================
# 12. 历史页
# ==================================================
with tab_history:
    for h in st.session_state.history:
        st.markdown(f"- **{h['label']}** ｜ +{h['points']} ｜ {h['time']}")
