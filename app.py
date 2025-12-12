import streamlit as st
import time
from PIL import Image
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import CLIPProcessor, CLIPModel
import json

# ==================================================
# 页面配置 - 沉浸式全屏体验
# ==================================================
st.set_page_config(
    page_title="SmartRecycle Pro AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 隐藏默认元素并美化界面 CSS
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none}
    .main {padding-top: 0rem;}
    h1, h2, h3 {text-align: center; font-family: 'Helvetica Neue', sans-serif;}
    
    /* 选项卡样式优化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 0 30px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 20px;
        background-color: #f1f5f9;
        border: none;
        margin: 0 5px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* 统计数字样式 */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    /* 去除图片上下的空白 */
    .stImage { margin-bottom: 0px; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Session State 初始化
# ==================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_points" not in st.session_state:
    st.session_state.total_points = 0
if "username" not in st.session_state:
    st.session_state.username = "Green Hero"
if "lang" not in st.session_state:
    st.session_state.lang = "kr"  # 默认韩语

# ==================================================
# 多语言字典 (已修复流程卡片混合语言的问题)
# ==================================================
TRANSLATIONS = {
    "zh": {
        "app_name": "智能分类 AI",
        "home": "🏠 首页",
        "scan": "📸 识别",
        "stats": "📊 数据",
        "profile": "👤 我的",
        "hero_title": "AI 助力垃圾分类",
        "hero_subtitle": "精准识别 · 积分奖励 · 守护地球",
        "upload_title": "拍摄或上传照片",
        "upload_btn": "📂 选择相册",
        "camera_btn": "📷 拍照",
        "scan_btn": "⚡ 立即识别",
        "points_display": "环保积分",
        "level": "等级",
        "congrats": "太棒了！",
        "earned": "获得奖励",
        "total_scans": "累计识别",
        "category_dist": "分类占比",
        "recent_activity": "最近动态",
        "username": "用户昵称",
        "save": "保存设置",
        "low_conf": "⚠️ AI 有点不确定，建议靠近一点再拍",
        "no_data": "暂无数据，快去识别第一件垃圾吧！",
        # --- 修复部分：流程步骤翻译 ---
        "step1_title": "1. 拍照上传",
        "step1_desc": "上传或拍摄垃圾照片",
        "step2_title": "2. AI 识别",
        "step2_desc": "智能分析垃圾类型",
        "step3_title": "3. 赚取积分",
        "step3_desc": "分类正确获得奖励",
    },
    "en": {
        "app_name": "SmartRecycle AI",
        "home": "🏠 Home",
        "scan": "📸 Scan",
        "stats": "📊 Stats",
        "profile": "👤 Profile",
        "hero_title": "AI Powered Recycling",
        "hero_subtitle": "Precision Scan · Earn Points · Save Earth",
        "upload_title": "Capture or Upload",
        "upload_btn": "📂 Gallery",
        "camera_btn": "📷 Camera",
        "scan_btn": "⚡ Identify Now",
        "points_display": "Eco Points",
        "level": "Level",
        "congrats": "Awesome!",
        "earned": "You Earned",
        "total_scans": "Total Scans",
        "category_dist": "Distribution",
        "recent_activity": "Recent Activity",
        "username": "Username",
        "save": "Save Changes",
        "low_conf": "⚠️ Low confidence. Try moving closer.",
        "no_data": "No data yet. Start scanning now!",
        # --- 修复部分：流程步骤翻译 ---
        "step1_title": "1. Capture",
        "step1_desc": "Take or Upload Photo",
        "step2_title": "2. Analyze",
        "step2_desc": "AI Identifies Waste",
        "step3_title": "3. Reward",
        "step3_desc": "Earn Eco Points",
    },
    "kr": {
        "app_name": "스마트 리사이클 AI",
        "home": "🏠 홈",
        "scan": "📸 스캔",
        "stats": "📊 통계",
        "profile": "👤 내 정보",
        "hero_title": "AI로 더 쉬운 분리수거",
        "hero_subtitle": "정확한 인식 · 포인트 적립 · 지구 보호",
        "upload_title": "사진 촬영 또는 업로드",
        "upload_btn": "📂 앨범 선택",
        "camera_btn": "📷 카메라",
        "scan_btn": "⚡ 분석 시작",
        "points_display": "에코 포인트",
        "level": "레벨",
        "congrats": "훌륭해요!",
        "earned": "획득",
        "total_scans": "총 스캔",
        "category_dist": "분류 통계",
        "recent_activity": "최근 활동",
        "username": "닉네임",
        "save": "저장",
        "low_conf": "⚠️ AI가 확실하지 않습니다. 더 가까이서 찍어주세요.",
        "no_data": "데이터가 없습니다. 첫 스캔을 시작해보세요!",
        # --- 修复部分：流程步骤翻译 ---
        "step1_title": "1. 촬영/업로드",
        "step1_desc": "사진 촬영 또는 앨범 업로드",
        "step2_title": "2. AI 분석",
        "step2_desc": "지능형 쓰레기 분석",
        "step3_title": "3. 보상 받기",
        "step3_desc": "에코 포인트 적립",
    }
}

# ==================================================
# 模型加载 (CLIP Model)
# ==================================================
@st.cache_resource
def load_model():
    """
    加载 OpenAI CLIP 模型。
    CLIP 擅长 Zero-Shot Classification，能通过文本描述更准确地识别物体。
    """
    try:
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        model.eval()  # 设置为评估模式
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

processor, model = load_model()

# ==================================================
# 类别定义 & 颜色配置
# ==================================================
CATEGORY_INFO = {
    "plastic": {
        "name": {"zh": "塑料", "en": "Plastic", "kr": "플라스틱"},
        "icon": "🥤", "color": "#10b981", "points": 10
    },
    "paper": {
        "name": {"zh": "纸类", "en": "Paper", "kr": "종이"},
        "icon": "📰", "color": "#f59e0b", "points": 5
    },
    "metal": {
        "name": {"zh": "金属", "en": "Metal", "kr": "금속"},
        "icon": "🥫", "color": "#3b82f6", "points": 15
    },
    "glass": {
        "name": {"zh": "玻璃", "en": "Glass", "kr": "유리"},
        "icon": "🍾", "color": "#a855f7", "points": 10
    },
    "cardboard": {
        "name": {"zh": "纸板", "en": "Cardboard", "kr": "골판지"},
        "icon": "📦", "color": "#d97706", "points": 8
    },
    "trash": {
        "name": {"zh": "一般垃圾", "en": "Trash", "kr": "일반쓰레기"},
        "icon": "🗑️", "color": "#64748b", "points": 2
    },
    "unknown": {
        "name": {"zh": "未知物体", "en": "Unknown", "kr": "알 수 없음"},
        "icon": "❓", "color": "#94a3b8", "points": 0
    }
}

# ==================================================
# AI 分类逻辑 (CLIP + 裁剪)
# ==================================================
def classify_image(image):
    """
    1. 自动裁剪图片中心 (去除背景干扰)
    2. 使用 CLIP 进行文本-图像匹配
    """
    # 1. 图像预处理 (中心裁剪)
    width, height = image.size
    # 取短边的 85% 作为裁剪区域
    new_size = min(width, height) * 0.85
    
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    
    cropped_image = image.crop((left, top, right, bottom))
    
    # 2. 定义 CLIP 提示词 (Prompt Engineering)
    # 顺序必须与下面的 labels 列表一一对应
    labels = ["plastic", "paper", "metal", "glass", "cardboard", "trash"]
    
    # 使用详细的英文描述，CLIP 对英文理解最好
    choices = [
        "a photo of plastic object, water bottle, plastic bag, or container",  # plastic
        "a photo of paper waste, newspaper, document, or white paper",         # paper
        "a photo of metal object, tin can, soda can, or aluminum foil",        # metal
        "a photo of glass bottle, glass jar, or broken glass",                 # glass
        "a photo of cardboard box, brown packaging box, or courier box",       # cardboard
        "a photo of general trash, food waste, dirty napkins, or mixed garbage" # trash
    ]
    
    # 3. 模型推理
    inputs = processor(
        text=choices, 
        images=cropped_image, 
        return_tensors="pt", 
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 计算概率
    logits_per_image = outputs.logits_per_image 
    probs = logits_per_image.softmax(dim=1) 
    score, idx = torch.max(probs, dim=-1)
    
    label = labels[idx.item()]
    confidence = score.item()
    
    # 4. 阈值过滤
    if confidence < 0.35:
        return "unknown", confidence
        
    return label, confidence

# ==================================================
# UI 渲染
# ==================================================

# --- 顶部导航栏 ---
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    lang_options = {"zh": "🇨🇳 中文", "en": "🇺🇸 English", "kr": "🇰🇷 한국어"}
    selected_lang = st.selectbox(
        "Language",
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(st.session_state.lang),
        key="lang_selector",
        label_visibility="collapsed"
    )
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

# 获取当前语言包
t = TRANSLATIONS[st.session_state.lang]

with col_center:
    st.markdown(f"<h2 style='text-align:center;margin:0;color:#0f172a;'>{t['app_name']}</h2>", unsafe_allow_html=True)

with col_right:
    level = st.session_state.total_points // 100 + 1
    st.markdown(f"""
    <div style='text-align:right; line-height:1.2;'>
        <span style='font-size:0.8rem;color:#64748b;'>{t['points_display']}</span><br>
        <span style='font-size:1.5rem;font-weight:700;color:#10b981;'>⭐ {st.session_state.total_points}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 主导航 Tab ---
tab1, tab2, tab3, tab4 = st.tabs([t["home"], t["scan"], t["stats"], t["profile"]])

# ==================================================
# Tab 1: 首页 (已修复)
# ==================================================
with tab1:
    st.markdown(f"""
    <div style='text-align:center;padding:50px 20px;
    background:linear-gradient(135deg,#d1fae5 0%, #a7f3d0 100%);
    border-radius:24px;margin-bottom:30px;box-shadow:0 10px 25px -5px rgba(16, 185, 129, 0.2);'>
        <h1 style='color:#065f46;margin-bottom:15px;'>{t['hero_title']}</h1>
        <p style='font-size:1.2rem;color:#047857;'>{t['hero_subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # 修复：使用翻译变量而不是写死文字
    steps = [
        ("📸", t['step1_title'], t['step1_desc']),
        ("🧠", t['step2_title'], t['step2_desc']),
        ("🎁", t['step3_title'], t['step3_desc'])
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3], steps):
        with col:
            st.markdown(f"""
            <div style='text-align:center;padding:20px;background:#f8fafc;border-radius:16px;border:1px solid #e2e8f0;'>
                <div style='font-size:2.5rem;margin-bottom:10px;'>{icon}</div>
                <div style='font-weight:bold;color:#334155;'>{title}</div>
                <div style='font-size:0.8rem;color:#94a3b8;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ==================================================
# Tab 2: 扫描页面
# ==================================================
with tab2:
    st.markdown(f"<h3 style='margin-bottom:20px;'>{t['upload_title']}</h3>", unsafe_allow_html=True)
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.info(f"📂 {t['upload_btn']}")
        uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
    with col_input2:
        st.info(f"📷 {t['camera_btn']}")
        camera_file = st.camera_input("Camera", label_visibility="collapsed")

    # 优先使用相机，其次使用上传
    image_source = camera_file if camera_file else uploaded_file
    
    if image_source:
        image = Image.open(image_source).convert("RGB")
        
        # 显示预览图
        st.image(image, caption="Preview", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 识别按钮
        if st.button(t['scan_btn'], type="primary", use_container_width=True):
            if not processor or not model:
                st.error("Model not loaded correctly.")
            else:
                with st.spinner("AI analyzing..."):
                    time.sleep(0.8) # 模拟延迟
                    
                    # === 调用核心分类函数 ===
                    label, confidence = classify_image(image)
                    # =======================
                    
                    cat_info = CATEGORY_INFO.get(label, CATEGORY_INFO["unknown"])
                    
                    points_earned = 0
                    if label != "unknown":
                        points_earned = cat_info["points"]
                        st.session_state.total_points += points_earned
                        
                        # 记录历史
                        st.session_state.history.insert(0, {
                            "label": cat_info["name"][st.session_state.lang],
                            "label_key": label,
                            "points": points_earned,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "confidence": confidence
                        })
                        
                        st.balloons()
                    
                    # 结果展示卡片
                    color = cat_info["color"]
                    cat_name = cat_info["name"][st.session_state.lang]
                    
                    st.markdown(f"""
                    <div style='margin-top:20px;padding:30px;border-radius:20px;
                    background:linear-gradient(135deg, {color}15, {color}05);
                    border:2px solid {color};text-align:center;'>
                        <div style='font-size:5rem;margin-bottom:10px;'>{cat_info['icon']}</div>
                        <h2 style='color:{color};margin:0;'>{cat_name}</h2>
                        <div style='font-size:2.5rem;font-weight:800;color:{color};margin:15px 0;'>
                            +{points_earned} PTS
                        </div>
                        <p style='color:#64748b;'>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if label == "unknown":
                        st.warning(t['low_conf'])
                    elif confidence < 0.5:
                        st.info(t['low_conf'])

# ==================================================
# Tab 3: 统计页面
# ==================================================
with tab3:
    if not st.session_state.history:
        st.info(t['no_data'])
    else:
        # 顶部三个指标
        c1, c2, c3 = st.columns(3)
        c1.metric(t['total_scans'], len(st.session_state.history))
        c2.metric(t['points_display'], st.session_state.total_points)
        c3.metric(t['level'], st.session_state.total_points // 100 + 1)
        
        st.markdown("---")
        
        # 环形图：分类占比
        st.markdown(f"#### {t['category_dist']}")
        
        counts = {}
        for h in st.session_state.history:
            key = h.get("label_key", "trash")
            counts[key] = counts.get(key, 0) + 1

        labels_display = []
        values = []
        colors = []
        
        for key, count in counts.items():
            if key in CATEGORY_INFO:
                labels_display.append(CATEGORY_INFO[key]["name"][st.session_state.lang])
                values.append(count)
                colors.append(CATEGORY_INFO[key]["color"])
        
        fig = go.Figure(data=[go.Pie(
            labels=labels_display, 
            values=values, 
            hole=0.6,
            marker=dict(colors=colors)
        )])
        fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # 最近记录列表
        st.markdown(f"#### {t['recent_activity']}")
        for item in st.session_state.history[:5]:
            with st.container():
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                padding:12px;margin-bottom:8px;background:white;border-radius:12px;border:1px solid #f1f5f9;'>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        <span style='font-size:1.2rem;'>♻️</span>
                        <div>
                            <div style='font-weight:bold;color:#334155;'>{item['label']}</div>
                            <div style='font-size:0.8rem;color:#94a3b8;'>{item['time']}</div>
                        </div>
                    </div>
                    <div style='font-weight:bold;color:#10b981;'>+{item['points']}</div>
                </div>
                """, unsafe_allow_html=True)

# ==================================================
# Tab 4: 个人资料
# ==================================================
with tab4:
    st.markdown(f"""
    <div style='text-align:center;padding:30px;
    background:linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    border-radius:20px;color:white;margin-bottom:30px;'>
        <div style='font-size:4rem;margin-bottom:10px;filter:drop-shadow(0 4px 6px rgba(0,0,0,0.2));'>😎</div>
        <h2 style='color:white;margin:0;'>{st.session_state.username}</h2>
        <p style='opacity:0.9;'>User ID: 8829103</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"#### ⚙️ {t['username']}")
    new_name = st.text_input("Edit Username", value=st.session_state.username, label_visibility="collapsed")
    if new_name != st.session_state.username:
        st.session_state.username = new_name
        st.success(t['save'])
        st.rerun()

    st.markdown("---")
    
    st.markdown("#### 🏆 Badges")
    b1, b2, b3 = st.columns(3)
    
    pts = st.session_state.total_points
    
    def badge_html(emoji, title, required, current):
        is_unlocked = current >= required
        opacity = "1" if is_unlocked else "0.4"
        grayscale = "0" if is_unlocked else "100%"
        status = "✅" if is_unlocked else f"🔒 {required}"
        return f"""
        <div style='text-align:center;opacity:{opacity};filter:grayscale({grayscale});'>
            <div style='font-size:3rem;'>{emoji}</div>
            <div style='font-weight:bold;font-size:0.9rem;margin-top:5px;'>{title}</div>
            <div style='font-size:0.8rem;color:#64748b;'>{status}</div>
        </div>
        """
        
    with b1: st.markdown(badge_html("🌱", "Starter", 50, pts), unsafe_allow_html=True)
    with b2: st.markdown(badge_html("🌿", "Expert", 200, pts), unsafe_allow_html=True)
    with b3: st.markdown(badge_html("🌳", "Master", 500, pts), unsafe_allow_html=True)
