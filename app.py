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
# 3. 多语言文案
# ==================================================
TRANS = {
    "zh": {
        "home": "系统主页",
        "scan": "智能识别",
        "data": "数据看板",
        "history": "历史记录",
        "upload": "上传图片",
        "start": "开始识别",
        "analyzing": "AI 正在分析中…",
        "result": "AI 建议分类",
        "points": "获得积分",
        "welcome": "上传或拍摄一张垃圾图片，AI 将给出分类建议。",
        "low_conf": "识别置信度较低，仅供参考"
    },
    "en": {
        "home": "Home",
        "scan": "AI Scan",
        "data": "Analytics",
        "history": "History",
        "upload": "Upload Image",
        "start": "Start Scan",
        "analyzing": "AI is analyzing…",
        "result": "AI Suggested Category",
        "points": "Points Earned",
        "welcome": "Upload a photo and let AI suggest a waste category.",
        "low_conf": "Low confidence, for reference only"
    },
    "kr": {
        "home": "홈",
        "scan": "AI 인식",
        "data": "데이터",
        "history": "기록",
        "upload": "이미지 업로드",
        "start": "스캔 시작",
        "analyzing": "AI 분석 중…",
        "result": "AI 분류 제안",
        "points": "획득 포인트",
        "welcome": "쓰레기 사진을 업로드하면 AI가 분류를 제안합니다.",
        "low_conf": "신뢰도가 낮아 참고용입니다"
    }
}

# ==================================================
# 4. 加载垃圾分类模型（稳定版）
# ==================================================
@st.cache_resource
def load_garbage_model():
    MODEL_NAME = "yangy50/garbage-classification"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

processor, model = load_garbage_model()

# ==================================================
# 5. id2label → UI 映射（核心）
# ==================================================
# ⚠️ key 必须与 model.config.id2label 完全一致
WASTE_UI = {
    "plastic": {
        "zh": ("塑料", "清洗后放入塑料回收桶", 10, "🥤", "#10b981"),
        "en": ("Plastic", "Clean and recycle as plastic", 10, "🥤", "#10b981"),
        "kr": ("플라스틱", "세척 후 플라스틱 수거함", 10, "🥤", "#10b981")
    },
    "paper": {
        "zh": ("纸类", "保持干燥后作为纸类回收", 5, "📰", "#f59e0b"),
        "en": ("Paper", "Keep dry and recycle as paper", 5, "📰", "#f59e0b"),
        "kr": ("종이", "물기 제거 후 종이 수거함", 5, "📰", "#f59e0b")
    },
    "metal": {
        "zh": ("金属", "压扁后放入金属回收桶", 15, "🥫", "#3b82f6"),
        "en": ("Metal", "Crush and recycle as metal", 15, "🥫", "#3b82f6"),
        "kr": ("금속", "압축 후 금属 수거함", 15, "🥫", "#3b82f6")
    },
    "glass": {
        "zh": ("玻璃", "小心放入玻璃回收桶", 10, "🍾", "#a855f7"),
        "en": ("Glass", "Handle carefully and recycle as glass", 10, "🍾", "#a855f7"),
        "kr": ("유리", "깨지지 않게 유리 수거함", 10, "🍾", "#a855f7")
    },
    "cardboard": {
        "zh": ("纸板", "压平后作为纸类回收", 5, "📦", "#f59e0b"),
        "en": ("Cardboard", "Flatten and recycle as paper", 5, "📦", "#f59e0b"),
        "kr": ("골판지", "펴서 종이류로 배출", 5, "📦", "#f59e0b")
    },
    "trash": {
        "zh": ("一般垃圾", "作为一般垃圾处理", 1, "🗑️", "#64748b"),
        "en": ("Trash", "Dispose as general waste", 1, "🗑️", "#64748b"),
        "kr": ("일반 쓰레기", "종량제 봉투 배출", 1, "🗑️", "#64748b")
    },
    "unknown": {
        "zh": ("无法识别", "图片不清晰，请人工判断", 0, "❓", "#94a3b8"),
        "en": ("Uncertain", "Low confidence, please classify manually", 0, "❓", "#94a3b8"),
        "kr": ("인식 불가", "확신 부족, 직접 분류해주세요", 0, "❓", "#94a3b8")
    }
}

# ==================================================
# 6. 分类函数（严格基于 id2label）
# ==================================================
def classify_waste(image, lang):
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    score, pred_id = torch.max(probs, dim=-1)

    score = score.item()
    label_key = model.config.id2label[pred_id.item()]

    # 置信度阈值
    if score < 0.35:
        label_key = "unknown"

    label, advice, points, icon, color = WASTE_UI[label_key][lang]

    return {
        "label": label,
        "advice": advice,
        "points": points,
        "icon": icon,
        "color": color,
        "score": score,
        "key": label_key
    }

# ==================================================
# 7. 侧边栏
# ==================================================
with st.sidebar:
    lang = st.selectbox(
        "Language / 언어",
        ["zh", "en", "kr"],
        format_func=lambda x: {"zh": "🇨🇳 中文", "en": "🇺🇸 English", "kr": "🇰🇷 한국어"}[x]
    )
    t = TRANS[lang]

    st.metric("⭐ Points", st.session_state.total_points)

    page = st.radio(
        "Navigation",
        [t["home"], t["scan"], t["data"], t["history"]],
        label_visibility="collapsed"
    )

# ==================================================
# 8. 页面逻辑
# ==================================================
if page == t["home"]:
    st.title("♻️ SmartRecycle")
    st.info(t["welcome"])

elif page == t["scan"]:
    st.title(f"📸 {t['scan']}")
    file = st.file_uploader(t["upload"], type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, width=320)

        if st.button(t["start"], use_container_width=True):
            with st.spinner(t["analyzing"]):
                time.sleep(1)

            res = classify_waste(img, lang)

            st.session_state.total_points += res["points"]
            st.session_state.last_res = res
            st.session_state.history.insert(0, {
                "label": res["label"],
                "key": res["key"],
                "points": res["points"],
                "time": datetime.now().strftime("%H:%M")
            })

    if st.session_state.last_res:
        r = st.session_state.last_res
        st.divider()
        st.subheader(t["result"])
        st.success(f"{r['icon']} {r['label']}")
        st.info(r["advice"])
        st.metric(t["points"], f"+{r['points']}")

        if r["score"] < 0.5:
            st.caption("⚠️ " + t["low_conf"])

elif page == t["data"]:
    st.title(f"📊 {t['data']}")

    if st.session_state.history:
        counter = {}
        for h in st.session_state.history:
            counter[h["label"]] = counter.get(h["label"], 0) + 1

        fig = px.pie(names=counter.keys(), values=counter.values(), hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

elif page == t["history"]:
    st.title(f"📜 {t['history']}")
    for h in st.session_state.history:
        st.markdown(f"- **{h['label']}** ｜ +{h['points']} ｜ {h['time']}")
