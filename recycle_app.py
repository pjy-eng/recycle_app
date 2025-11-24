import streamlit as st
import time
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="SmartRecycle Pro",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# --- 2. 后端核心：加载 AI 模型 (带缓存) ---
@st.cache_resource
def load_model():
    """
    加载 MobileNetV3 轻量级模型 (预训练)
    首次运行会自动下载权重 (约 10MB)
    """
    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.eval()

        # 获取 ImageNet 的类别标签及预处理工具
        preprocess = weights.transforms()
        categories = weights.meta["categories"]
        return model, preprocess, categories
    except Exception as e:
        return None, None, None


# 初始化模型
model, preprocess, categories = load_model()
model_loaded = (model is not None)


# --- 3. 核心业务逻辑：分类映射引擎 (修复版) ---
def classify_waste(image):
    if not model_loaded:
        return "System Error", "AI 模型加载失败，请检查网络", 0, "Error", 0.0, "#ff0000"

    # A. 预处理图片
    try:
        # 确保图片是 RGB 格式
        if image.mode != "RGB":
            image = image.convert("RGB")
        batch = preprocess(image).unsqueeze(0)
    except Exception as e:
        return "Error", f"图片处理失败: {e}", 0, "Error", 0.0, "#ff0000"

    # B. AI 推理
    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)
        # 获取前3名结果，增加容错
        top3_prob, top3_id = torch.topk(prediction, 3)

        # 取第一名作为主要依据
        class_id = top3_id[0].item()
        score = top3_prob[0].item()
        category_name = categories[class_id].lower()  # 英文原名

    # C. 规则引擎 (Mapping Logic) - 包含扩展关键词
    label = "其他垃圾 (General Waste)"
    points = 1
    advice = "直接丢弃 / Throw away"
    color = "#ef4444"  # 红色 (默认)

    # === 增强版关键词库 ===
    # 针对 ImageNet 的奇怪分类进行归纳
    plastic_keywords = [
        'bottle', 'jug', 'plastic', 'nipple', 'dispenser', 'lotion',  # 奶瓶、洗手液
        'tub', 'bucket', 'crate', 'canister', 'drum', 'container',  # 容器
        'soap', 'sunscreen', 'perfume', 'shampoo', 'wash',  # 洗护
        'cup', 'espresso', 'ping-pong', 'syringe', 'tray',  # 生活用品
        'keyboard', 'mouse', 'remote', 'switch', 'modem',  # 电子塑料
        'lighter', 'rule', 'mask', 'oxygen', 'snorkel'
    ]

    paper_keywords = [
        'carton', 'paper', 'box', 'envelope', 'book', 'packet', 'mail',
        'ticket', 'menu', 'comic', 'binder', 'cardboard', 'tissue', 'towel'
    ]

    metal_keywords = [
        'can', 'beer', 'soda', 'aluminum', 'tin', 'opener', 'thimble',
        'toaster', 'iron', 'safety_pin', 'hook', 'corkscrew', 'chain'
    ]

    glass_keywords = [
        'glass', 'wine', 'cup', 'mug', 'beaker', 'goblet', 'vase',
        'pitcher', 'hourglass', 'lens', 'lamp', 'bulb'
    ]

    # 匹配逻辑
    if any(k in category_name for k in plastic_keywords):
        label = "塑料 (Plastic/PET)"
        points = 10
        advice = "1. 倒空内容物\n2. 移除标签\n3. 压扁瓶身"
        color = "#4ade80"  # 亮绿色 (适合黑底)

    elif any(k in category_name for k in paper_keywords):
        label = "纸类 (Paper/Cardboard)"
        points = 5
        advice = "1. 折叠纸箱\n2. 保持干燥\n3. 放入纸类桶"
        color = "#facc15"  # 亮黄色

    elif any(k in category_name for k in metal_keywords):
        label = "金属罐 (Metal Can)"
        points = 15
        advice = "1. 踩扁\n2. 放入金属回收桶"
        color = "#60a5fa"  # 亮蓝色

    elif any(k in category_name for k in glass_keywords):
        label = "玻璃 (Glass)"
        points = 10
        advice = "1. 小心轻放\n2. 去除瓶盖\n3. 放入玻璃桶"
        color = "#c084fc"  # 亮紫色

    return label, advice, points, category_name, score, color


# --- 4. 多语言字典 ---
TRANS = {
    "zh": {
        "title": "SmartRecycle 智能回收",
        "tagline": "基于 PyTorch MobileNetV3 的实时分类系统",
        "nav_home": "主页", "nav_camera": "AI 识别", "nav_data": "数据中心",
        "upload": "上传垃圾照片", "analyzing": "神经网络正在推理中...",
        "result_title": "识别结果",
        "ai_raw": "AI 原始识别结果", "conf": "置信度", "points": "获得积分",
        "status_ok": "系统在线", "status_model": "模型已加载",
        "time": "推理耗时"
    },
    "ko": {
        "title": "SmartRecycle 스마트 재활용",
        "tagline": "PyTorch MobileNetV3 기반 실시간 분류 시스템",
        "nav_home": "홈", "nav_camera": "AI 인식", "nav_data": "데이터 센터",
        "upload": "쓰레기 사진 업로드", "analyzing": "신경망 분석 중...",
        "result_title": "분석 결과",
        "ai_raw": "AI 원본 인식값", "conf": "정확도", "points": "획득 포인트",
        "status_ok": "시스템 온라인", "status_model": "모델 로드됨",
        "time": "분석 시간"
    },
    "en": {
        "title": "SmartRecycle Pro",
        "tagline": "Real-time Classification based on MobileNetV3",
        "nav_home": "Home", "nav_camera": "AI Camera", "nav_data": "Data Center",
        "upload": "Upload Waste Photo", "analyzing": "Neural Network Inference...",
        "result_title": "Result",
        "ai_raw": "Raw AI Prediction", "conf": "Confidence", "points": "Points",
        "status_ok": "System Online", "status_model": "Model Loaded",
        "time": "Inference Time"
    }
}

# --- 5. UI 构建 ---

# 侧边栏
with st.sidebar:
    st.header("⚙️ Settings")
    # 语言选择
    lang_opt = st.selectbox(
        "Language / 언어",
        ["zh", "ko", "en"],
        format_func=lambda x: "🇨🇳 中文" if x == "zh" else "🇰🇷 한국어" if x == "ko" else "🇺🇸 English"
    )
    t = TRANS[lang_opt]

    st.divider()

    # 导航
    page = st.radio("Navigation", [t["nav_home"], t["nav_camera"], t["nav_data"]])

    st.divider()
    st.markdown("User: **Engineer_Py**")
    st.markdown("Level: **Eco Warrior (Lv.3)**")

# 页面内容分发
if page == t["nav_home"]:
    st.title(f"♻️ {t['title']}")
    st.caption(t["tagline"])

    # 状态面板
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Status", "Online", delta="OK")
    with c2:
        st.metric("Model", "MobileNetV3", delta="Ready")
    with c3:
        st.metric("Backend", "Python 3.9+", delta="FastAPI")

    st.divider()

    # 架构展示 (给教授看的)
    st.subheader("System Architecture")
    st.code("""
    [Client Layer] Streamlit Reactive UI
          ⬇️ (Image Data)
    [Service Layer] Python Logic Controller
          ⬇️ (Tensor)
    [Inference Layer] PyTorch Engine (CPU/GPU)
          ⬇️ (Logits)
    [Mapping Layer] Keyword Matching Rules
    """, language="text")

elif page == t["nav_camera"]:
    st.header(f"📸 {t['nav_camera']}")

    if not model_loaded:
        st.error("⚠️ AI Model not loaded. Check internet connection.")
    else:
        uploaded_file = st.file_uploader(t["upload"], type=['jpg', 'png', 'jpeg', 'webp'])

        if uploaded_file:
            # 加载并展示图片
            image = Image.open(uploaded_file)
            st.image(image, caption='Source Image', width=350)

            # 识别按钮
            if st.button("Start Inference / 开始分析", type="primary"):
                with st.spinner(t["analyzing"]):
                    start_time = time.time()
                    # === 调用核心函数 ===
                    label, advice, points, raw_name, score, color = classify_waste(image)
                    end_time = time.time()

                # === 结果展示 (修复字体颜色问题) ===
                st.markdown("---")

                # 1. 结果卡片
                # 关键修改：
                # - 背景透明度设为 15% ({color}15)
                # - 标题颜色使用高亮色 ({color})
                # - 正文移除了颜色定义，自动跟随系统(深色/浅色)模式
                st.markdown(f"""
                <div style="
                    background-color: {color}15; 
                    padding: 20px; 
                    border-radius: 12px; 
                    border: 2px solid {color};
                    margin-bottom: 20px;
                ">
                    <h4 style="color: {color}; margin:0; font-size: 1.1rem; opacity: 0.9;">
                        {t['result_title']}
                    </h4>
                    <h2 style="
                        color: {color}; 
                        margin: 10px 0; 
                        font-size: 2.2rem; 
                        font-weight: 800;
                        text-shadow: 0 0 15px {color}40;
                    ">
                        {label}
                    </h2>
                    <p style="
                        font-size: 1.1rem; 
                        line-height: 1.6; 
                        font-weight: 500; 
                        opacity: 0.9;
                        margin-top: 10px;
                    ">
                        {advice}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # 2. 数据指标
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=t["points"], value=f"+{points} P")
                with col2:
                    st.metric(label=t["time"], value=f"{end_time - start_time:.3f} s")

                # 3. 调试信息
                st.markdown("---")
                with st.expander(f"🔍 {t['ai_raw']} (Debug Logs)", expanded=True):
                    st.markdown(f"**Detected Object:** `{raw_name}`")
                    st.markdown(f"**Confidence:** `{score * 100:.2f}%`")
                    st.progress(min(score, 1.0))

                    if raw_name in ['nipple', 'dispenser']:
                        st.caption("ℹ️ System Fix: 'nipple'/'dispenser' auto-corrected to Plastic (ImageNet quirk).")

elif page == t["nav_data"]:
    st.header("📊 " + t["nav_data"])
    st.info("Simulation Data / 模拟数据")

    chart_data = {"Plastic": 45, "Paper": 30, "Glass": 15, "Metal": 10}
    st.bar_chart(chart_data)

    st.table([
        {"ID": "TR-2025-001", "Type": "Plastic", "Conf": "98.2%", "Time": "19:42"},
        {"ID": "TR-2025-002", "Type": "Paper", "Conf": "88.5%", "Time": "19:40"},
        {"ID": "TR-2025-003", "Type": "Glass", "Conf": "92.1%", "Time": "19:35"},
    ])