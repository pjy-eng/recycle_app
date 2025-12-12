import streamlit as st
import time
from PIL import Image
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import json

# ==================================================
# 页面配置 - 去除侧边栏
# ==================================================
st.set_page_config(
    page_title="SmartRecycle Pro",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 隐藏侧边栏
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none}
    .main {padding-top: 0rem;}
    h1, h2, h3 {text-align: center;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 40px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 12px 12px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .upload-section {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Session State
# ==================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_points" not in st.session_state:
    st.session_state.total_points = 0
if "username" not in st.session_state:
    st.session_state.username = "环保达人"
if "lang" not in st.session_state:
    st.session_state.lang = "kr"  # 默认韩语

# ==================================================
# 多语言 - 修复逻辑
# ==================================================
TRANSLATIONS = {
    "zh": {
        "app_name": "智能分类",
        "home": "🏠 首页",
        "scan": "📸 扫描",
        "stats": "📊 统计",
        "profile": "👤 我的",
        "hero_title": "让垃圾分类变得简单有趣",
        "hero_subtitle": "用AI识别，赚积分，保护地球",
        "upload_title": "上传或拍摄垃圾照片",
        "upload_btn": "📂 选择图片",
        "camera_btn": "📷 打开相机",
        "scan_btn": "🔍 开始识别",
        "points_display": "我的积分",
        "level": "等级",
        "congrats": "太棒了！",
        "earned": "获得",
        "total_scans": "总扫描次数",
        "category_dist": "分类统计",
        "recent_activity": "最近记录",
        "username": "昵称",
        "save": "保存",
        "low_conf": "⚠️ 置信度较低，建议重新拍摄",
        "no_data": "还没有数据，快去扫描吧！",
    },
    "en": {
        "app_name": "SmartRecycle",
        "home": "🏠 Home",
        "scan": "📸 Scan",
        "stats": "📊 Stats",
        "profile": "👤 Profile",
        "hero_title": "Make Recycling Easy & Fun",
        "hero_subtitle": "Scan with AI, Earn Points, Save Earth",
        "upload_title": "Upload or Capture Waste Photo",
        "upload_btn": "📂 Choose Image",
        "camera_btn": "📷 Open Camera",
        "scan_btn": "🔍 Start Scan",
        "points_display": "My Points",
        "level": "Level",
        "congrats": "Awesome!",
        "earned": "Earned",
        "total_scans": "Total Scans",
        "category_dist": "Category Distribution",
        "recent_activity": "Recent Activity",
        "username": "Username",
        "save": "Save",
        "low_conf": "⚠️ Low confidence, please retake photo",
        "no_data": "No data yet. Start scanning!",
    },
    "kr": {
        "app_name": "스마트리사이클",
        "home": "🏠 홈",
        "scan": "📸 스캔",
        "stats": "📊 통계",
        "profile": "👤 프로필",
        "hero_title": "쉽고 재미있는 분리수거",
        "hero_subtitle": "AI 인식, 포인트 적립, 지구 보호",
        "upload_title": "쓰레기 사진 업로드 또는 촬영",
        "upload_btn": "📂 이미지 선택",
        "camera_btn": "📷 카메라 열기",
        "scan_btn": "🔍 인식 시작",
        "points_display": "내 포인트",
        "level": "레벨",
        "congrats": "훌륭해요!",
        "earned": "획득",
        "total_scans": "총 스캔 횟수",
        "category_dist": "분류 통계",
        "recent_activity": "최근 기록",
        "username": "닉네임",
        "save": "저장",
        "low_conf": "⚠️ 신뢰도가 낮습니다. 다시 촬영해주세요",
        "no_data": "아직 데이터가 없습니다. 스캔을 시작하세요!",
    }
}

# ==================================================
# 模型加载
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
# 垃圾分类标签映射
# ==================================================
CATEGORY_INFO = {
    "plastic": {
        "name": {"zh": "塑料", "en": "Plastic", "kr": "플라스틱"},
        "icon": "🥤",
        "color": "#10b981",
        "points": 10
    },
    "paper": {
        "name": {"zh": "纸类", "en": "Paper", "kr": "종이"},
        "icon": "📰",
        "color": "#f59e0b",
        "points": 5
    },
    "metal": {
        "name": {"zh": "金属", "en": "Metal", "kr": "금속"},
        "icon": "🥫",
        "color": "#3b82f6",
        "points": 15
    },
    "glass": {
        "name": {"zh": "玻璃", "en": "Glass", "kr": "유리"},
        "icon": "🍾",
        "color": "#a855f7",
        "points": 10
    },
    "cardboard": {
        "name": {"zh": "纸板", "en": "Cardboard", "kr": "골판지"},
        "icon": "📦",
        "color": "#f59e0b",
        "points": 5
    },
    "trash": {
        "name": {"zh": "一般垃圾", "en": "Trash", "kr": "일반쓰레기"},
        "icon": "🗑️",
        "color": "#64748b",
        "points": 1
    },
    "unknown": {
        "name": {"zh": "未知", "en": "Unknown", "kr": "알 수 없음"},
        "icon": "❓",
        "color": "#94a3b8",
        "points": 0
    }
}

# ==================================================
# AI 分类函数 - 提高准确度
# ==================================================
def classify_image(image):
    """单张图片分类，使用更严格的阈值"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    score, idx = torch.max(probs, dim=-1)
    
    label = model.config.id2label[idx.item()]
    confidence = score.item()
    
    # 更严格的置信度阈值
    if confidence < 0.45:
        label = "unknown"
    
    return label, confidence

# ==================================================
# 顶部导航栏 - 语言选择器 + 积分显示
# ==================================================
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    lang_options = {
        "zh": "🇨🇳 中文",
        "en": "🇺🇸 English", 
        "kr": "🇰🇷 한국어"
    }
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

with col_center:
    t = TRANSLATIONS[st.session_state.lang]
    st.markdown(f"<h1 style='text-align:center;margin:0;'>♻️ {t['app_name']}</h1>", 
                unsafe_allow_html=True)

with col_right:
    level = st.session_state.total_points // 100 + 1
    st.markdown(f"""
    <div style='text-align:right;'>
        <div style='font-size:0.9rem;color:#64748b;'>{t['points_display']}</div>
        <div style='font-size:2rem;font-weight:700;color:#10b981;'>
            ⭐ {st.session_state.total_points}
        </div>
        <div style='font-size:0.8rem;color:#94a3b8;'>{t['level']} {level}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================================================
# 主导航 - 大按钮样式
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs([
    t["home"],
    t["scan"], 
    t["stats"],
    t["profile"]
])

# ==================================================
# Tab 1: 首页 - Hero Section
# ==================================================
with tab1:
    st.markdown(f"""
    <div style='text-align:center;padding:60px 20px;
    background:linear-gradient(135deg,#10b98120,#059669 20);border-radius:24px;'>
        <h1 style='font-size:3rem;margin-bottom:20px;'>🌍 {t['hero_title']}</h1>
        <p style='font-size:1.5rem;color:#64748b;'>{t['hero_subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 使用说明 - 简洁版
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align:center;padding:30px;background:#f8fafc;border-radius:16px;'>
            <div style='font-size:3rem;'>📸</div>
            <h3>1. 拍照 / 上传</h3>
            <p style='color:#64748b;'>清晰拍摄垃圾照片</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align:center;padding:30px;background:#f8fafc;border-radius:16px;'>
            <div style='font-size:3rem;'>🤖</div>
            <h3>2. AI 识别</h3>
            <p style='color:#64748b;'>智能判断垃圾类型</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align:center;padding:30px;background:#f8fafc;border-radius:16px;'>
            <div style='font-size:3rem;'>⭐</div>
            <h3>3. 赚积分</h3>
            <p style='color:#64748b;'>正确分类获得奖励</p>
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# Tab 2: 扫描页面 - 核心功能
# ==================================================
with tab2:
    st.markdown(f"<h2>📸 {t['upload_title']}</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    upload_image = None
    camera_image = None
    
    with col1:
        st.markdown(f"### {t['upload_btn']}")
        uploaded_file = st.file_uploader(
            "upload",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            upload_image = Image.open(uploaded_file).convert("RGB")
            st.image(upload_image, use_container_width=True)
    
    with col2:
        st.markdown(f"### {t['camera_btn']}")
        # 相机输入 - 放大显示
        camera_photo = st.camera_input(
            "camera",
            label_visibility="collapsed"
        )
        if camera_photo:
            camera_image = Image.open(camera_photo).convert("RGB")
    
    # 选择要识别的图片
    selected_image = upload_image or camera_image
    
    if selected_image:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 大按钮开始识别
        if st.button(f"🔍 {t['scan_btn']}", 
                     use_container_width=True, 
                     type="primary",
                     key="scan_button"):
            
            with st.spinner("🤖 AI 正在分析中..."):
                time.sleep(1.5)  # 增加仪式感
                
                label, confidence = classify_image(selected_image)
                
                # 获取分类信息
                category = CATEGORY_INFO[label]
                name = category["name"][st.session_state.lang]
                icon = category["icon"]
                color = category["color"]
                points = category["points"]
                
                # 更新积分和历史
                st.session_state.total_points += points
                st.session_state.history.insert(0, {
                    "label": name,
                    "points": points,
                    "time": datetime.now().strftime("%H:%M"),
                    "confidence": confidence
                })
            
            # 🎉 庆祝动画
            st.balloons()
            
            # 结果展示 - 大卡片
            st.markdown(f"""
            <div style='margin:30px auto;max-width:500px;padding:40px;
            border-radius:24px;text-align:center;
            background:linear-gradient(135deg,{color}40,{color}10);
            border:3px solid {color};box-shadow:0 10px 40px {color}40;'>
                <div style='font-size:6rem;margin-bottom:20px;'>{icon}</div>
                <h2 style='font-size:2.5rem;margin:20px 0;'>{name}</h2>
                <div style='font-size:3rem;font-weight:700;color:{color};margin:20px 0;'>
                    +{points} {t['points_display']}
                </div>
                <div style='font-size:1rem;color:#64748b;'>
                    {t['congrats']} {t['earned']} {points} {t['points_display']}!
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 低置信度警告
            if confidence < 0.6:
                st.warning(t['low_conf'])
            
            # 显示置信度
            st.progress(confidence, text=f"置信度: {confidence:.1%}")

# ==================================================
# Tab 3: 统计页面
# ==================================================
with tab3:
    st.markdown(f"<h2>📊 {t['stats']}</h2>", unsafe_allow_html=True)
    
    if st.session_state.history:
        # 总览指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                t['total_scans'],
                len(st.session_state.history),
                delta="+1" if len(st.session_state.history) > 0 else None
            )
        
        with col2:
            st.metric(
                t['points_display'],
                st.session_state.total_points,
                delta=f"+{st.session_state.history[0]['points']}" if st.session_state.history else None
            )
        
        with col3:
            level = st.session_state.total_points // 100 + 1
            next_level_points = (level * 100) - st.session_state.total_points
            st.metric(
                t['level'],
                level,
                delta=f"{next_level_points} to next"
            )
        
        st.markdown("---")
        
        # 分类统计饼图
        st.markdown(f"### {t['category_dist']}")
        
        category_count = {}
        for record in st.session_state.history:
            label = record["label"]
            category_count[label] = category_count.get(label, 0) + 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(category_count.keys()),
            values=list(category_count.values()),
            hole=0.4,
            marker=dict(
                colors=['#10b981', '#f59e0b', '#3b82f6', '#a855f7', '#64748b']
            )
        )])
        
        fig.update_layout(
            showlegend=True,
            height=400,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 最近记录
        st.markdown(f"### {t['recent_activity']}")
        
        for i, record in enumerate(st.session_state.history[:10]):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{record['label']}**")
            with col2:
                st.markdown(f"+{record['points']} pts")
            with col3:
                st.markdown(f"`{record['time']}`")
            
            if i < len(st.session_state.history) - 1:
                st.markdown("<hr style='margin:5px 0;'>", unsafe_allow_html=True)
    
    else:
        st.info(t['no_data'])

# ==================================================
# Tab 4: 个人资料
# ==================================================
with tab4:
    st.markdown(f"<h2>👤 {t['profile']}</h2>", unsafe_allow_html=True)
    
    # 用户信息卡片
    st.markdown(f"""
    <div style='text-align:center;padding:40px;background:linear-gradient(135deg,#10b98120,#05966920);
    border-radius:24px;margin:20px 0;'>
        <div style='font-size:5rem;margin-bottom:20px;'>👤</div>
        <h2>{st.session_state.username}</h2>
        <p style='color:#64748b;font-size:1.2rem;'>
            {t['level']} {st.session_state.total_points // 100 + 1} | 
            {st.session_state.total_points} {t['points_display']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 修改昵称
    st.markdown(f"### ✏️ {t['username']}")
    new_username = st.text_input(
        "username_input",
        value=st.session_state.username,
        label_visibility="collapsed"
    )
    
    if st.button(f"💾 {t['save']}", use_container_width=True):
        st.session_state.username = new_username
        st.success("✅ 保存成功！")
    
    st.markdown("---")
    
    # 等级进度
    level = st.session_state.total_points // 100 + 1
    progress = (st.session_state.total_points % 100) / 100
    
    st.markdown(f"### 🎯 {t['level']}进度")
    st.progress(progress)
    st.caption(f"还需 {100 - (st.session_state.total_points % 100)} 积分升到 Level {level + 1}")
    
    # 成就徽章（示例）
    st.markdown("### 🏆 成就")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.total_points >= 50:
            st.markdown("🥉 **新手** ✅")
        else:
            st.markdown("🔒 新手 (需要50积分)")
    
    with col2:
        if st.session_state.total_points >= 200:
            st.markdown("🥈 **达人** ✅")
        else:
            st.markdown("🔒 达人 (需要200积分)")
    
    with col3:
        if st.session_state.total_points >= 500:
            st.markdown("🥇 **大师** ✅")
        else:
            st.markdown("🔒 大师 (需要500积分)")
