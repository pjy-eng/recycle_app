import streamlit as st
import time
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime, timedelta
import plotly.graph_objects as go
import torch
from transformers import CLIPProcessor, CLIPModel
import random

# ==================================================
# 1. 页面配置 (必须在最前面)
# ==================================================
st.set_page_config(
    page_title="EcoScan AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# 2. 全局样式 (CSS) - 包含 Radio 模拟 Tab 的样式
# ==================================================
st.markdown("""
<style>
    /* 隐藏默认元素 */
    [data-testid="collapsedControl"] {display: none}
    .main {padding: 0; max-width: 100%;}
    
    /* 字体优化 */
    * {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;}
    
    /* 导航栏样式 */
    .nav-container {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid #e2e8f0;
        padding: 12px 0;
        margin-bottom: 0;
    }

    /* =========================================
       自定义导航条 (模拟 Tabs)
       ========================================= */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        background: transparent;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 0;
        width: 100%;
        margin-bottom: 20px;
    }
    div[role="radiogroup"] > div {
        display: flex;
        gap: 0;
        width: auto;
    }
    div[role="radiogroup"] label {
        background: transparent !important;
        border: none;
        padding: 10px 32px;
        border-radius: 0;
        transition: all 0.2s;
        margin: 0;
        color: #94a3b8;
        font-weight: 500;
        font-size: 15px;
        cursor: pointer;
    }
    /* 选中状态 */
    div[role="radiogroup"] label[data-checked="true"] {
        border-bottom: 3px solid #10b981 !important;
        color: #10b981 !important;
        font-weight: bold;
    }
    div[role="radiogroup"] label:hover {
        color: #10b981;
    }
    /* 隐藏单选圆圈 */
    div[role="radiogroup"] label > div:first-child {
        display: none;
    }

    /* 按钮优化 */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
    }
    
    /* 度量指标美化 */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* 进度条 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 10px;
    }
    
    /* 图片预览 */
    .stImage {margin: 0;}
    
    /* 输入框 */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 12px;
    }
    
    /* 移除多余间距 */
    .element-container {margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. Session State 初始化
# ==================================================
def init_session_state():
    defaults = {
        "history": [],
        "total_points": 0,
        "username": "EcoWarrior",
        "lang": "zh", 
        "first_visit": True,
        "streak_days": 0,
        "last_scan_date": None,
        "total_co2_saved": 0,
        "achievements": [],
        "scan_mode": "instant",
        "onboarding_done": False,
        "current_tab": None, # 新增：控制当前Tab
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================================================
# 4. 多语言配置
# ==================================================
TRANSLATIONS = {
    "zh": {
        "app_name": "EcoScan AI",
        "tagline": "AI 助力环保，让分类更简单",
        "nav_home": "🏠 首页",
        "nav_scan": "📸 扫描",
        "nav_insights": "📊 洞察",
        "nav_profile": "👤 我的",
        "hero_title": "拍照识别，智能分类",
        "hero_subtitle": "每次正确分类，都在守护地球",
        "get_started": "开始扫描",
        "upload_photo": "📤 上传照片",
        "take_photo": "📷 拍照",
        "instant_scan": "⚡ 即时扫描",
        "batch_scan": "📦 批量扫描",
        "analyzing": "AI 分析中",
        "result_title": "识别结果",
        "confidence": "置信度",
        "earned_points": "获得积分",
        "co2_saved": "减少碳排放",
        "scan_another": "继续扫描",
        "view_history": "查看记录",
        "low_confidence_title": "AI 不太确定",
        "low_confidence_msg": "照片可能模糊或物体不清晰，试试这些建议：",
        "tip_lighting": "💡 确保光线充足",
        "tip_focus": "🎯 对准物体中心",
        "tip_distance": "📏 保持适当距离",
        "help_us_learn": "帮助 AI 学习",
        "correct_category": "正确分类是？",
        "submit_feedback": "提交反馈",
        "thanks_feedback": "感谢！您的反馈让 AI 更聪明",
        "total_scans": "总扫描",
        "eco_score": "环保分数",
        "current_level": "当前等级",
        "streak": "连续天数",
        "category_breakdown": "分类分布",
        "recent_activity": "最近活动",
        "eco_impact": "环保影响",
        "trees_planted": "相当于种树",
        "water_saved": "节约用水",
        "achievements": "成就徽章",
        "locked": "未解锁",
        "profile_settings": "个人设置",
        "username": "用户昵称",
        "language": "语言",
        "save_changes": "保存更改",
        "badge_beginner": "入门者",
        "badge_explorer": "探索者",
        "badge_expert": "专家",
        "badge_master": "大师",
        "badge_legend": "传奇",
        "badge_streak": "连续王",
        "badge_variety": "全能手",
        "no_data": "还没有数据，开始第一次扫描吧！",
        "welcome_title": "欢迎来到 EcoScan",
        "welcome_msg": "让我们一起用 AI 让垃圾分类变简单",
        "onboard_step1": "拍照或上传图片",
        "onboard_step2": "AI 识别垃圾类型",
        "onboard_step3": "获得积分和环保成就",
        "skip": "跳过",
        "next": "下一步",
        "start": "开始体验",
    },
    "en": {
        "app_name": "EcoScan AI",
        "tagline": "AI-Powered Recycling Made Simple",
        "nav_home": "🏠 Home",
        "nav_scan": "📸 Scan",
        "nav_insights": "📊 Insights",
        "nav_profile": "👤 Profile",
        "hero_title": "Snap. Scan. Sort.",
        "hero_subtitle": "Every correct sort saves our planet",
        "get_started": "Start Scanning",
        "upload_photo": "📤 Upload Photo",
        "take_photo": "📷 Take Photo",
        "instant_scan": "⚡ Instant Scan",
        "batch_scan": "📦 Batch Scan",
        "analyzing": "AI Analyzing",
        "result_title": "Scan Result",
        "confidence": "Confidence",
        "earned_points": "Points Earned",
        "co2_saved": "CO₂ Reduced",
        "scan_another": "Scan Another",
        "view_history": "View History",
        "low_confidence_title": "AI is Uncertain",
        "low_confidence_msg": "Photo might be blurry or unclear. Try these tips:",
        "tip_lighting": "💡 Ensure good lighting",
        "tip_focus": "🎯 Focus on object center",
        "tip_distance": "📏 Keep proper distance",
        "help_us_learn": "Help AI Learn",
        "correct_category": "Correct category?",
        "submit_feedback": "Submit Feedback",
        "thanks_feedback": "Thanks! Your feedback makes AI smarter",
        "total_scans": "Total Scans",
        "eco_score": "Eco Score",
        "current_level": "Current Level",
        "streak": "Day Streak",
        "category_breakdown": "Category Breakdown",
        "recent_activity": "Recent Activity",
        "eco_impact": "Eco Impact",
        "trees_planted": "Trees Equivalent",
        "water_saved": "Water Saved",
        "achievements": "Achievements",
        "locked": "Locked",
        "profile_settings": "Profile Settings",
        "username": "Username",
        "language": "Language",
        "save_changes": "Save Changes",
        "badge_beginner": "Beginner",
        "badge_explorer": "Explorer",
        "badge_expert": "Expert",
        "badge_master": "Master",
        "badge_legend": "Legend",
        "badge_streak": "Streak King",
        "badge_variety": "All-Rounder",
        "no_data": "No data yet. Start your first scan!",
        "welcome_title": "Welcome to EcoScan",
        "welcome_msg": "Let's make recycling simple with AI",
        "onboard_step1": "Snap or upload photo",
        "onboard_step2": "AI identifies waste type",
        "onboard_step3": "Earn points & eco badges",
        "skip": "Skip",
        "next": "Next",
        "start": "Get Started",
    },
    "kr": {
        "app_name": "EcoScan AI",
        "tagline": "AI로 더 쉬운 분리수거",
        "nav_home": "🏠 홈",
        "nav_scan": "📸 스캔",
        "nav_insights": "📊 인사이트",
        "nav_profile": "👤 프로필",
        "hero_title": "촬영. 인식. 분류.",
        "hero_subtitle": "올바른 분류는 지구를 지킵니다",
        "get_started": "스캔 시작",
        "upload_photo": "📤 사진 업로드",
        "take_photo": "📷 사진 촬영",
        "instant_scan": "⚡ 즉시 스캔",
        "batch_scan": "📦 배치 스캔",
        "analyzing": "AI 분석 중",
        "result_title": "스캔 결과",
        "confidence": "신뢰도",
        "earned_points": "획득 포인트",
        "co2_saved": "CO₂ 감소",
        "scan_another": "계속 스캔",
        "view_history": "기록 보기",
        "low_confidence_title": "AI가 확신하지 못합니다",
        "low_confidence_msg": "사진이 흐릿하거나 불명확할 수 있습니다. 다음을 시도해보세요:",
        "tip_lighting": "💡 충분한 조명 확보",
        "tip_focus": "🎯 물체 중앙에 초점",
        "tip_distance": "📏 적절한 거리 유지",
        "help_us_learn": "AI 학습 돕기",
        "correct_category": "올바른 분류는?",
        "submit_feedback": "피드백 제출",
        "thanks_feedback": "감사합니다! 피드백으로 AI가 더 똑똑해집니다",
        "total_scans": "총 스캔",
        "eco_score": "에코 점수",
        "current_level": "현재 레벨",
        "streak": "연속 일수",
        "category_breakdown": "카테고리 분포",
        "recent_activity": "최근 활동",
        "eco_impact": "환경 영향",
        "trees_planted": "나무 심기 효과",
        "water_saved": "절약한 물",
        "achievements": "성취 배지",
        "locked": "잠김",
        "profile_settings": "프로필 설정",
        "username": "사용자 이름",
        "language": "언어",
        "save_changes": "변경사항 저장",
        "badge_beginner": "초보자",
        "badge_explorer": "탐험가",
        "badge_expert": "전문가",
        "badge_master": "마스터",
        "badge_legend": "전설",
        "badge_streak": "연속왕",
        "badge_variety": "올라운더",
        "no_data": "데이터가 없습니다. 첫 스캔을 시작하세요!",
        "welcome_title": "EcoScan에 오신 것을 환영합니다",
        "welcome_msg": "AI로 분리수거를 쉽게 만들어요",
        "onboard_step1": "사진 촬영 또는 업로드",
        "onboard_step2": "AI가 쓰레기 유형 인식",
        "onboard_step3": "포인트와 환경 배지 획득",
        "skip": "건너뛰기",
        "next": "다음",
        "start": "시작하기",
    }
}

# ==================================================
# 5. 分类与数据配置
# ==================================================
CATEGORIES = {
    "plastic": {
        "name": {"zh": "塑料", "en": "Plastic", "kr": "플라스틱"},
        "icon": "🥤",
        "color": "#10b981",
        "points": 10,
        "co2_kg": 0.5,
        "prompts": ["plastic bottle", "plastic container", "plastic waste"]
    },
    "paper": {
        "name": {"zh": "纸张", "en": "Paper", "kr": "종이"},
        "icon": "📄",
        "color": "#f59e0b",
        "points": 5,
        "co2_kg": 0.3,
        "prompts": ["paper waste", "newspaper", "white paper"]
    },
    "cardboard": {
        "name": {"zh": "纸板", "en": "Cardboard", "kr": "골판지"},
        "icon": "📦",
        "color": "#d97706",
        "points": 8,
        "co2_kg": 0.4,
        "prompts": ["cardboard box", "cardboard waste"]
    },
    "metal": {
        "name": {"zh": "金属", "en": "Metal", "kr": "금속"},
        "icon": "🥫",
        "color": "#3b82f6",
        "points": 15,
        "co2_kg": 0.8,
        "prompts": ["metal can", "aluminum can", "tin can"]
    },
    "glass": {
        "name": {"zh": "玻璃", "en": "Glass", "kr": "유리"},
        "icon": "🍾",
        "color": "#a855f7",
        "points": 12,
        "co2_kg": 0.6,
        "prompts": ["glass bottle", "glass jar"]
    },
    "trash": {
        "name": {"zh": "一般垃圾", "en": "General Trash", "kr": "일반쓰레기"},
        "icon": "🗑️",
        "color": "#64748b",
        "points": 2,
        "co2_kg": 0.1,
        "prompts": ["general trash", "food waste", "garbage"]
    },
}

# ==================================================
# 6. AI 模型与推理逻辑
# ==================================================
@st.cache_resource
def load_clip_model():
    try:
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

processor, model = load_clip_model()

def preprocess_image(image):
    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    image = image.filter(ImageFilter.SHARPEN)
    return image

def classify_image(image):
    if not processor or not model:
        return "trash", 0.0
    
    processed_image = preprocess_image(image)
    
    category_keys = list(CATEGORIES.keys())
    prompts = []
    for key in category_keys:
        prompts.append(f"a photo of {random.choice(CATEGORIES[key]['prompts'])}")
    
    inputs = processor(
        text=prompts,
        images=processed_image,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    confidence, idx = torch.max(probs, dim=-1)
    predicted_category = category_keys[idx.item()]
    confidence_value = confidence.item()
    
    if confidence_value < 0.35:
        return "trash", confidence_value
    
    return predicted_category, confidence_value

# ==================================================
# 7. 业务逻辑函数
# ==================================================
def update_streak():
    today = datetime.now().date()
    if st.session_state.last_scan_date is None:
        st.session_state.streak_days = 1
    else:
        last_date = st.session_state.last_scan_date
        days_diff = (today - last_date).days
        if days_diff == 1:
            st.session_state.streak_days += 1
        elif days_diff > 1:
            st.session_state.streak_days = 1
    st.session_state.last_scan_date = today

def check_achievements():
    points = st.session_state.total_points
    scans = len(st.session_state.history)
    streak = st.session_state.streak_days
    achievements = st.session_state.achievements
    
    if scans >= 1 and "beginner" not in achievements: achievements.append("beginner")
    if scans >= 10 and "explorer" not in achievements: achievements.append("explorer")
    if scans >= 50 and "expert" not in achievements: achievements.append("expert")
    if scans >= 100 and "master" not in achievements: achievements.append("master")
    if scans >= 500 and "legend" not in achievements: achievements.append("legend")
    if streak >= 7 and "streak" not in achievements: achievements.append("streak")
    
    if st.session_state.history:
        unique_categories = set([h["category"] for h in st.session_state.history])
        if len(unique_categories) >= 4 and "variety" not in achievements:
            achievements.append("variety")

def add_scan_record(category, confidence, points, co2):
    record = {
        "category": category,
        "confidence": confidence,
        "points": points,
        "co2": co2,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    st.session_state.history.insert(0, record)
    st.session_state.total_points += points
    st.session_state.total_co2_saved += co2
    update_streak()
    check_achievements()

def get_level():
    return st.session_state.total_points // 100 + 1

def get_level_progress():
    return (st.session_state.total_points % 100) / 100

# ==================================================
# 8. UI 组件
# ==================================================
def render_navbar(t):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        lang_options = {"zh": "🇨🇳 中文", "en": "🇺🇸 English", "kr": "🇰🇷 한국어"}
        selected_lang = st.selectbox(
            "Language",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=list(lang_options.keys()).index(st.session_state.lang),
            label_visibility="collapsed"
        )
        if selected_lang != st.session_state.lang:
            st.session_state.lang = selected_lang
            st.rerun()
    with col2:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<h2 style='margin:0;color:#0f172a;font-weight:800;'>🌱 {t['app_name']}</h2>"
            f"<p style='margin:0;color:#64748b;font-size:0.85rem;'>{t['tagline']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col3:
        level = get_level()
        st.markdown(
            f"<div style='text-align:right;'>"
            f"<div style='font-size:0.75rem;color:#64748b;'>Lv.{level}</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:#10b981;'>"
            f"⭐ {st.session_state.total_points}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

def render_onboarding(t):
    if not st.session_state.onboarding_done:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#d1fae5,#a7f3d0);padding:60px 40px;border-radius:24px;text-align:center;margin-bottom:30px;'>"
            f"<h1 style='color:#065f46;margin-bottom:20px;'>{t['welcome_title']}</h1>"
            f"<p style='font-size:1.3rem;color:#047857;margin-bottom:40px;'>{t['welcome_msg']}</p>"
            f"</div>", unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns(3)
        steps = [("📸", t['onboard_step1']), ("🤖", t['onboard_step2']), ("🎁", t['onboard_step3'])]
        for col, (icon, text) in zip([col1, col2, col3], steps):
            with col:
                st.markdown(f"<div style='text-align:center;padding:30px;background:white;border-radius:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);'><div style='font-size:3.5rem;margin-bottom:15px;'>{icon}</div><p style='font-size:1.1rem;color:#334155;font-weight:500;'>{text}</p></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col_skip, col_start = st.columns(2)
        with col_skip:
            if st.button(t['skip'], use_container_width=True):
                st.session_state.onboarding_done = True
                st.rerun()
        with col_start:
            if st.button(t['start'], use_container_width=True, type="primary"):
                st.session_state.onboarding_done = True
                st.rerun()
        return True
    return False

def render_scan_result(t, category, confidence, image):
    cat_info = CATEGORIES[category]
    name = cat_info["name"][st.session_state.lang]
    icon = cat_info["icon"]
    color = cat_info["color"]
    points = cat_info["points"]
    co2 = cat_info["co2_kg"]
    
    st.balloons()
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{color}15,{color}05);border:3px solid {color};border-radius:24px;padding:40px;text-align:center;margin:30px 0;box-shadow:0 10px 40px {color}30;'>"
        f"<div style='font-size:6rem;margin-bottom:20px;'>{icon}</div>"
        f"<h2 style='color:{color};font-size:2.5rem;margin:15px 0;'>{name}</h2>"
        f"<div style='display:flex;justify-content:center;gap:40px;margin:30px 0;'>"
        f"<div><span style='color:#64748b;'>🎯 {t['confidence']}</span><br><span style='font-size:1.8rem;font-weight:700;color:{color};'>{confidence:.0%}</span></div>"
        f"<div><span style='color:#64748b;'>⭐ {t['earned_points']}</span><br><span style='font-size:1.8rem;font-weight:700;color:{color};'>+{points}</span></div>"
        f"<div><span style='color:#64748b;'>🌍 {t['co2_saved']}</span><br><span style='font-size:1.8rem;font-weight:700;color:{color};'>{co2:.1f}kg</span></div>"
        f"</div></div>", unsafe_allow_html=True
    )
    
    if confidence < 0.6:
        with st.expander(f"⚠️ {t['low_confidence_title']}", expanded=True):
            st.warning(t['low_confidence_msg'])
            st.markdown(f"- {t['tip_lighting']}")
            st.markdown(f"- {t['tip_focus']}")
            st.markdown(f"- {t['tip_distance']}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"📸 {t['scan_another']}", use_container_width=True):
            st.session_state.pop('current_image', None)
            st.rerun()
    with col2:
        if st.button(f"📊 {t['view_history']}", use_container_width=True):
            # 跳转到洞察页
            st.session_state.current_tab = TRANSLATIONS[st.session_state.lang]['nav_insights']
            st.rerun()

# ==================================================
# 9. 主程序结构 (包含修复后的跳转逻辑)
# ==================================================
def main():
    t = TRANSLATIONS[st.session_state.lang]
    render_navbar(t)
    if render_onboarding(t): return
    
    # -----------------------------------------------
    # 核心修改：使用 Radio + Session State 替代 st.tabs
    # -----------------------------------------------
    tabs_options = [t['nav_home'], t['nav_scan'], t['nav_insights'], t['nav_profile']]
    
    # 确保 session_state 初始化
    if st.session_state.current_tab not in tabs_options:
        st.session_state.current_tab = tabs_options[0]

    # 跳转回调函数
    def go_to_scan_tab():
        st.session_state.current_tab = t['nav_scan']

    # 导航栏 (CSS将其样式化为Tab)
    selected_tab = st.radio(
        "", 
        options=tabs_options, 
        horizontal=True, 
        label_visibility="collapsed",
        key="current_tab" # 双向绑定
    )

    # --- Tab 1: 首页 ---
    if selected_tab == t['nav_home']:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#10b98115,#05966915);padding:80px 40px;border-radius:24px;text-align:center;margin-bottom:40px;'>"
            f"<h1 style='font-size:3.5rem;color:#065f46;margin-bottom:20px;'>{t['hero_title']}</h1>"
            f"<p style='font-size:1.5rem;color:#047857;margin-bottom:40px;'>{t['hero_subtitle']}</p>"
            f"</div>", unsafe_allow_html=True
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric(t['total_scans'], len(st.session_state.history))
        with col2: st.metric(t['eco_score'], st.session_state.total_points)
        with col3: st.metric(t['current_level'], get_level())
        with col4: st.metric(f"{t['streak']} 🔥", st.session_state.streak_days)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 按钮绑定回调函数，实现跳转
        st.button(f"📸 {t['get_started']}", use_container_width=True, type="primary", on_click=go_to_scan_tab)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if st.session_state.history:
            st.markdown(f"### 🌍 {t['eco_impact']}")
            trees = st.session_state.total_co2_saved / 20
            water = st.session_state.total_points * 2
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='background:linear-gradient(135deg,#d1fae5,#a7f3d0);padding:30px;border-radius:16px;text-align:center;'><div style='font-size:3rem;'>🌳</div><h3 style='color:#065f46;'>{trees:.1f}</h3><p style='color:#047857;'>{t['trees_planted']}</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='background:linear-gradient(135deg,#dbeafe,#bfdbfe);padding:30px;border-radius:16px;text-align:center;'><div style='font-size:3rem;'>💧</div><h3 style='color:#1e40af;'>{water:.0f}L</h3><p style='color:#1e3a8a;'>{t['water_saved']}</p></div>", unsafe_allow_html=True)

    # --- Tab 2: 扫描页面 ---
    elif selected_tab == t['nav_scan']:
        st.markdown(f"### 📸 {t['instant_scan']}")
        col1, col2 = st.columns([1, 1])
        img_file_buffer = None
        
        with col1:
            st.markdown(f"#### {t['upload_photo']}")
            uploaded_file = st.file_uploader("upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="uploader")
            if uploaded_file: img_file_buffer = uploaded_file
        with col2:
            st.markdown(f"#### {t['take_photo']}")
            camera_photo = st.camera_input("camera", label_visibility="collapsed", key="camera")
            if camera_photo: img_file_buffer = camera_photo
            
        if img_file_buffer:
            image = Image.open(img_file_buffer).convert("RGB")
            st.image(image, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button(f"⚡ {t['instant_scan']}", use_container_width=True, type="primary", key="scan_btn"):
                with st.spinner(f"🤖 {t['analyzing']}..."):
                    time.sleep(1.0)
                    category, confidence = classify_image(image)
                    cat_info = CATEGORIES[category]
                    add_scan_record(category, confidence, cat_info["points"], cat_info["co2_kg"])
                    render_scan_result(t, category, confidence, image)

    # --- Tab 3: 数据洞察 ---
    elif selected_tab == t['nav_insights']:
        if not st.session_state.history:
            st.info(t['no_data'])
        else:
            col1, col2, col3 = st.columns(3)
            with col1: st.metric(t['total_scans'], len(st.session_state.history))
            with col2: st.metric(t['eco_score'], st.session_state.total_points)
            with col3: 
                st.metric(t['current_level'], get_level())
                st.progress(get_level_progress())
            st.markdown("---")
            
            st.markdown(f"### 📊 {t['category_breakdown']}")
            category_counts = {}
            for record in st.session_state.history:
                cat = record['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            labels = [CATEGORIES[k]["name"][st.session_state.lang] for k in category_counts.keys()]
            values = list(category_counts.values())
            colors = [CATEGORIES[k]["color"] for k in category_counts.keys()]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker=dict(colors=colors), textinfo='label+percent', textfont=dict(size=14))])
            fig.update_layout(showlegend=False, height=400, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            
            st.markdown(f"### 🕐 {t['recent_activity']}")
            for record in st.session_state.history[:10]:
                cat_info = CATEGORIES[record['category']]
                name = cat_info["name"][st.session_state.lang]
                color = cat_info["color"]
                st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;padding:16px;margin-bottom:10px;background:white;border-radius:12px;border-left:4px solid {color};box-shadow:0 1px 3px rgba(0,0,0,0.05);'><div style='display:flex;align-items:center;gap:12px;'><span style='font-size:1.8rem;'>{cat_info['icon']}</span><div><div style='font-weight:600;color:#334155;'>{name}</div><div style='font-size:0.8rem;color:#94a3b8;'>{record['timestamp']}</div></div></div><div style='font-weight:700;color:{color};font-size:1.2rem;'>+{record['points']}</div></div>", unsafe_allow_html=True)

    # --- Tab 4: 个人资料 ---
    elif selected_tab == t['nav_profile']:
        level = get_level()
        st.markdown(f"<div style='background:linear-gradient(135deg,#4facfe,#00f2fe);padding:50px;border-radius:24px;text-align:center;color:white;margin-bottom:30px;'><div style='font-size:5rem;margin-bottom:20px;'>👤</div><h2 style='color:white;margin:0;'>{st.session_state.username}</h2><p style='opacity:0.9;font-size:1.2rem;margin-top:10px;'>Level {level} EcoWarrior</p></div>", unsafe_allow_html=True)
        
        st.markdown(f"### ⚙️ {t['profile_settings']}")
        new_username = st.text_input(t['username'], value=st.session_state.username, max_chars=20)
        if new_username != st.session_state.username:
            if st.button(t['save_changes'], type="primary"):
                st.session_state.username = new_username
                st.success("✅ Saved!")
                st.rerun()
        st.markdown("---")
        
        st.markdown(f"### 🏆 {t['achievements']}")
        achievements_config = {
            "beginner": ("🌱", t['badge_beginner']), "explorer": ("🔍", t['badge_explorer']),
            "expert": ("⚡", t['badge_expert']), "master": ("👑", t['badge_master']),
            "legend": ("🌟", t['badge_legend']), "streak": ("🔥", t['badge_streak']),
            "variety": ("🎨", t['badge_variety']),
        }
        cols = st.columns(4)
        for idx, (key, (icon, name)) in enumerate(achievements_config.items()):
            unlocked = key in st.session_state.achievements
            with cols[idx % 4]:
                opacity = "1" if unlocked else "0.3"
                filter_style = "" if unlocked else "filter:grayscale(100%);"
                st.markdown(f"<div style='text-align:center;padding:20px;background:#f8fafc;border-radius:16px;border:2px solid {'#10b981' if unlocked else '#e2e8f0'};opacity:{opacity};{filter_style}'><div style='font-size:3rem;margin-bottom:10px;'>{icon}</div><div style='font-weight:600;color:#334155;'>{name}</div><div style='font-size:0.75rem;color:#94a3b8;margin-top:5px;'>{'✅ Unlocked' if unlocked else f'🔒 {t['locked']}'}</div></div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown(f"### 📈 {t['current_level']} Progress")
        next_level_points = (level * 100) - (st.session_state.total_points % 100)
        st.progress(get_level_progress())
        st.caption(f"Next level in {next_level_points} points")

if __name__ == "__main__":
    main()
