import streamlit as st
import time
import requests
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from datetime import datetime
import plotly.express as px
from streamlit_lottie import st_lottie

# --- 1. 页面配置与 CSS ---
st.set_page_config(
    page_title="SmartRecycle Pro v2.1",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局字体与间距 */
    .main { padding: 1rem 2rem; }

    /* 标题动画 */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animated-title { animation: fadeInDown 0.8s ease-out; }

    /* 结果卡片 - 增加透明度适配深色背景 */
    .result-card {
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.3s;
    }
    .result-card:hover { transform: translateY(-5px); }

    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }

    /* 侧边栏 - 深色模式适配 */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #2e303e 0%, #1e1e24 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* 进度条颜色 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981, #3b82f6);
    }

    /* 历史记录条目 - 适配深色 */
    .history-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border-left: 5px solid #ddd;
        transition: transform 0.2s;
        color: white;
    }
    .history-item:hover { transform: scale(1.01); background: rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)


# --- 2. 辅助函数：加载资源 ---

@st.cache_resource
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None


lottie_scanning = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m64r7l.json")
lottie_eco = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_u4yrau.json")


@st.cache_resource
def load_model():
    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.eval()
        preprocess = weights.transforms()
        categories = weights.meta["categories"]
        return model, preprocess, categories
    except Exception as e:
        return None, None, None


model, preprocess, categories = load_model()
model_loaded = (model is not None)


# --- 3. 核心逻辑：多语言分类映射 ---
def classify_waste(image, lang="zh"):
    """
    现在这个函数接收 lang 参数，根据语言返回对应的建议
    """
    if not model_loaded:
        return "System Error", "Model Error", 0, "Error", 0.0, "#ff0000", "❌"

    try:
        if image.mode != "RGB": image = image.convert("RGB")
        batch = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
            top3_prob, top3_id = torch.topk(prediction, 3)
            class_id = top3_id[0].item()
            score = top3_prob[0].item()
            category_name = categories[class_id].lower()
    except:
        return "Error", "Process Error", 0, "Error", 0.0, "#ff0000", "❌"

    # --- 核心修改：文案数据库 ---
    # 格式: {key: {zh: ..., en: ..., kr: ...}}
    WASTE_INFO = {
        "plastic": {
            "zh": {"label": "塑料 (Plastic)", "advice": "💧 倒空液体 -> 压扁 -> 放入蓝色回收桶"},
            "en": {"label": "Plastic", "advice": "💧 Empty liquid -> Crush -> Blue Bin"},
            "kr": {"label": "플라스틱 (Plastic)", "advice": "💧 내용물 비우기 -> 압축하기 -> 플라스틱 수거함"}
        },
        "paper": {
            "zh": {"label": "纸类 (Paper)", "advice": "📄 保持干燥 -> 折叠平整 -> 放入纸类回收桶"},
            "en": {"label": "Paper", "advice": "📄 Keep dry -> Flatten -> Paper Bin"},
            "kr": {"label": "종이 (Paper)", "advice": "📄 물기 제거 -> 납작하게 펴기 -> 종이 수거함"}
        },
        "metal": {
            "zh": {"label": "金属 (Metal)", "advice": "🦶 踩扁罐体 -> 放入金属回收桶"},
            "en": {"label": "Metal", "advice": "🦶 Crush cans -> Metal Bin"},
            "kr": {"label": "금속 (Metal)", "advice": "🦶 캔 압축하기 -> 고철류 수거함"}
        },
        "glass": {
            "zh": {"label": "玻璃 (Glass)", "advice": "💥 小心轻放 -> 去盖 -> 放入玻璃回收桶"},
            "en": {"label": "Glass", "advice": "💥 Handle with care -> Remove lid -> Glass Bin"},
            "kr": {"label": "유리 (Glass)", "advice": "💥 깨지지 않게 주의 -> 뚜껑 제거 -> 유리 수거함"}
        },
        "general": {
            "zh": {"label": "其他垃圾 (General)", "advice": "🗑️ 无法识别具体分类，请作为一般垃圾处理"},
            "en": {"label": "General Waste", "advice": "🗑️ Unidentified. Dispose as general waste"},
            "kr": {"label": "일반 쓰레기 (General)", "advice": "🗑️ 분류 불가. 종량제 봉투에 버려주세요"}
        },
        "unknown": {
            "zh": {"label": "❓ 未知物体", "advice": "🤔 AI 感到困惑，建议人工分类"},
            "en": {"label": "❓ Unknown Object", "advice": "🤔 AI is confused. Please classify manually"},
            "kr": {"label": "❓ 알 수 없음", "advice": "🤔 AI가 인식하지 못했습니다. 직접 분류해주세요"}
        }
    }

    # 关键词映射
    plastic_keys = ['bottle', 'plastic', 'container', 'tub', 'cup', 'nipple', 'lotion']
    paper_keys = ['carton', 'paper', 'box', 'envelope', 'book', 'cardboard', 'tissue']
    metal_keys = ['can', 'aluminum', 'tin', 'beer', 'soda', 'iron']
    glass_keys = ['glass', 'wine', 'mug', 'goblet', 'vase', 'bulb']

    # 确定类别 Key
    type_key = "general"  # 默认
    color = "#64748b"
    icon = "🗑️"
    points = 1

    if score < 0.15:
        type_key = "unknown"
        points = 0
        color = "#94a3b8"
        icon = "❓"
    elif any(k in category_name for k in plastic_keys):
        type_key = "plastic"
        points = 10
        color = "#10b981"
        icon = "🥤"
    elif any(k in category_name for k in paper_keys):
        type_key = "paper"
        points = 5
        color = "#f59e0b"
        icon = "📰"
    elif any(k in category_name for k in metal_keys):
        type_key = "metal"
        points = 15
        color = "#3b82f6"
        icon = "🥫"
    elif any(k in category_name for k in glass_keys):
        type_key = "glass"
        points = 10
        color = "#a855f7"
        icon = "🍾"

    # 根据 lang 获取文本
    info = WASTE_INFO[type_key][lang]
    label = info["label"]
    advice = info["advice"]

    return label, advice, points, category_name, score, color, icon


# --- 4. Session State ---
if 'history' not in st.session_state: st.session_state.history = []
if 'total_points' not in st.session_state: st.session_state.total_points = 0
if 'classification_count' not in st.session_state: st.session_state.classification_count = 0

# --- 5. 多语言配置 (增加韩语) ---
TRANS = {
    "zh": {
        "nav_home": "🏠 系统主页", "nav_camera": "📸 智能识别", "nav_data": "📊 数据看板", "nav_history": "📜 历史档案",
        "tab_upload": "📂 上传照片", "tab_camera": "📷 实时拍照",
        "start_scan": "开始识别", "analyzing": "AI 正在思考中...",
        "feedback_title": "🛠️ 识别不准？", "feedback_btn": "提交修正",
        "toast_success": "识别成功！积分 +", "unknown": "未知",
        "level": "等级", "points_label": "累计积分", "welcome": "👋 欢迎！使用 AI 识别废弃物并获取积分。"
    },
    "en": {
        "nav_home": "🏠 Home", "nav_camera": "📸 AI Scan", "nav_data": "📊 Analytics", "nav_history": "📜 History",
        "tab_upload": "📂 Upload File", "tab_camera": "📷 Camera",
        "start_scan": "Start Scan", "analyzing": "AI is thinking...",
        "feedback_title": "🛠️ Wrong Result?", "feedback_btn": "Submit Fix",
        "toast_success": "Success! Points +", "unknown": "Unknown",
        "level": "Level", "points_label": "Total Points",
        "welcome": "👋 Welcome! Use AI Camera to scan and earn points."
    },
    "kr": {
        "nav_home": "🏠 홈", "nav_camera": "📸 AI 스캔", "nav_data": "📊 분석", "nav_history": "📜 기록",
        "tab_upload": "📂 사진 업로드", "tab_camera": "📷 카메라 촬영",
        "start_scan": "스캔 시작", "analyzing": "AI 분석 중...",
        "feedback_title": "🛠️ 결과가 틀렸나요?", "feedback_btn": "수정 제출",
        "toast_success": "성공! 포인트 +", "unknown": "알 수 없음",
        "level": "레벨", "points_label": "총 포인트", "welcome": "👋 환영합니다! AI 카메라로 쓰레기를 분류하고 포인트를 받으세요."
    }
}

# --- 6. 侧边栏 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299901.png", width=50)
    st.title("SmartRecycle")

    # 语言选择逻辑
    lang_options = {"zh": "🇨🇳 中文", "en": "🇺🇸 English", "kr": "🇰🇷 한국어"}
    lang_code = st.selectbox(
        "Language / 언어",
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x]
    )
    t = TRANS[lang_code]

    st.markdown("---")
    page = st.radio("Navigation", [t["nav_home"], t["nav_camera"], t["nav_data"], t["nav_history"]],
                    label_visibility="collapsed")

    st.markdown("---")
    st.metric(t["points_label"], st.session_state.total_points)
    st.progress(min(st.session_state.total_points / 500, 1.0))
    st.caption(f"🏆 {t['level']}: " + str(st.session_state.total_points // 100 + 1))

# --- 7. 主页面逻辑 ---

# === 🏠 主页 ===
if page == t["nav_home"]:
    st.markdown('<h1 class="animated-title">♻️ SmartRecycle Pro</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.info(t["welcome"])
        # 统计数据行
        m1, m2, m3 = st.columns(3)
        m1.metric("📸 Scans", st.session_state.classification_count)
        m2.metric("⭐ Points", st.session_state.total_points)
        m3.metric("🤖 Model", "V3-Small")

        st.markdown("### 🌟 Why Recycle?")
        # 这里的静态文本也可以根据 t 字典进行优化，这里暂略
        st.markdown("""
        * **Reduce Pollution** - 减少污染 / 오염 감소
        * **Conserve Resources** - 节约资源 / 자원 절약
        * **Earn Points** - 赚取积分 / 포인트 적립
        """)

    with col2:
        if lottie_eco:
            st_lottie(lottie_eco, height=250, key="eco_anim")

# === 📸 智能识别 (核心功能) ===
elif page == t["nav_camera"]:
    st.markdown(f'<h1 class="animated-title">{t["nav_camera"]}</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs([t["tab_upload"], t["tab_camera"]])
    img_input = None

    with tab1:
        uploaded_file = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
        if uploaded_file:
            img_input = Image.open(uploaded_file)
            st.image(img_input, caption="Preview", width=300)

    with tab2:
        camera_file = st.camera_input("Camera")
        if camera_file:
            img_input = Image.open(camera_file)

    if img_input:
        if st.button(t["start_scan"], type="primary", use_container_width=True):
            if lottie_scanning:
                with st.empty():
                    st_lottie(lottie_scanning, height=150, key="loading")
                    time.sleep(1.2)

            # --- 关键修改：将 lang_code 传入函数 ---
            start_t = time.time()
            label, advice, points, raw, score, color, icon = classify_waste(img_input, lang=lang_code)
            cost_t = time.time() - start_t

            st.session_state.classification_count += 1
            st.session_state.total_points += points
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M"),
                "label": label, "points": points, "conf": score, "color": color, "icon": icon
            })

            st.session_state.last_res = {
                "label": label, "advice": advice, "points": points,
                "score": score, "color": color, "icon": icon, "raw": raw, "time": cost_t
            }

            st.toast(f"{t['toast_success']} {points}", icon="🎉")
            st.rerun()

    if hasattr(st.session_state, 'last_res'):
        res = st.session_state.last_res

        st.divider()
        st.markdown(f"""
        <div class="result-card" style="border: 2px solid {res['color']}; background: linear-gradient(180deg, {res['color']}10 0%, rgba(255,255,255,0.05) 100%);">
            <div style="font-size: 5rem;">{res['icon']}</div>
            <h1 style="color: {res['color']}; margin: 0;">{res['label']}</h1>
            <p style="font-size: 1.2rem; opacity: 0.8; margin-top: 10px;">{res['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Conf", f"{res['score'] * 100:.1f}%")
        c2.metric("Time", f"{res['time']:.3f}s")
        c3.metric("XP", f"+{res['points']}")

        with st.expander(t["feedback_title"]):
            st.write(f"Raw Model: `{res['raw']}`")
            # 这里的选项也可以做国际化，为简单起见先保留
            user_correction = st.selectbox("Correct Type", ["Plastic", "Paper", "Metal", "Glass", "Other"])
            if st.button(t["feedback_btn"]):
                st.success("Feedback Recorded!")

# === 📊 数据看板 ===
elif page == t["nav_data"]:
    st.markdown(f'<h1 class="animated-title">{t["nav_data"]}</h1>', unsafe_allow_html=True)

    if st.session_state.history:
        col_chart, col_stats = st.columns([2, 1])
        with col_chart:
            st.markdown("### Distribution")
            data = {}
            for item in st.session_state.history:
                lbl = item['label'].split('(')[0].strip()
                data[lbl] = data.get(lbl, 0) + 1

            fig = px.pie(values=list(data.values()), names=list(data.keys()), hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.markdown("### Stats")
            st.metric("Total Scans", len(st.session_state.history))
            st.metric("Total XP", st.session_state.total_points)

# === 📜 历史档案 ===
elif page == t["nav_history"]:
    st.markdown(f'<h1 class="animated-title">{t["nav_history"]}</h1>', unsafe_allow_html=True)

    if st.button("🗑️ Clear"):
        st.session_state.history = []
        st.rerun()

    for item in st.session_state.history:
        st.markdown(f"""
        <div class="history-item" style="border-left-color: {item['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.5rem; margin-right: 15px;">{item['icon']}</span>
                    <span style="font-weight: bold; font-size: 1.1rem;">{item['label']}</span>
                    <span style="opacity: 0.7; font-size: 0.9rem; margin-left: 10px;">{item['time']}</span>
                </div>
                <div style="text-align: right;">
                    <div style="color: {item['color']}; font-weight: bold;">+{item['points']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)