import streamlit as st
import time
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
import plotly.graph_objects as go
import torch
from transformers import CLIPProcessor, CLIPModel
import random

# ==================================================
# 1. 页面配置 (必须在最前面)
# ==================================================
st.set_page_config(
    page_title="EcoScan KR",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# 2. 全局样式 (CSS)
# ==================================================
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none}
    .main {padding: 0; max-width: 100%;}
    * {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;}
    
    /* 导航栏样式优化 */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        background: transparent;
        border-bottom: 2px solid #f1f5f9;
        margin-bottom: 20px;
    }
    div[role="radiogroup"] label {
        background: transparent !important;
        border: none;
        padding: 10px 24px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 16px;
        cursor: pointer;
        transition: 0.3s;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        border-bottom: 3px solid #10b981 !important;
        color: #10b981 !important;
    }
    div[role="radiogroup"] label:hover {
        color: #10b981;
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* 徽章容器 */
    .badge-card {
        background: #f8fafc;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 2px solid #e2e8f0;
        transition: all 0.3s;
    }
    
    /* 消除图片默认边距 */
    .stImage {margin: 0;}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. Session State 初始化
# ==================================================
def init_session_state():
    defaults = {
        "history": [],
        "total_points": 0,
        "username": "EcoCitizen",
        "lang": "kr",  # 默认韩语
        "current_tab": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================================================
# 4. 严格的多语言字典 (确保不混合)
# ==================================================
TRANSLATIONS = {
    "kr": {
        "app_name": "에코스캔 AI",
        "tagline": "AI로 완벽한 분리수거",
        "nav_home": "🏠 홈", "nav_scan": "📸 스캔", "nav_insights": "📊 통계", "nav_profile": "👤 내 정보",
        
        # Hero Section
        "hero_title": "이 쓰레기, 어떻게 버리죠?",
        "hero_subtitle": "사진을 찍으면 올바른 분리배출 방법을 알려드립니다",
        
        # 步骤 (Steps) - 纯韩语
        "step1_title": "1. 촬영/업로드", "step1_desc": "쓰레기 사진을 찍으세요",
        "step2_title": "2. AI 분석", "step2_desc": "종류와 배출 방법을 확인하세요",
        "step3_title": "3. 포인트 획득", "step3_desc": "환경을 지키고 보상을 받으세요",
        
        # 快速指南 (Quick Guide)
        "guide_plastic": "플라스틱", "guide_plastic_desc": "헹구고 라벨 제거",
        "guide_vinyl": "비닐류", "guide_vinyl_desc": "깨끗한 상태로 배출",
        "guide_paper": "종이/박스", "guide_paper_desc": "펼쳐서 배출",
        "guide_trash": "일반쓰레기", "guide_trash_desc": "오염된 것은 여기로",
        "quick_guide_title": "📋 분리수거 핵심 가이드",

        # 功能按钮
        "upload_btn": "📂 사진 업로드", "camera_btn": "📷 카메라",
        "scan_action": "🔍 분석 시작",
        "analyzing": "AI가 분석 중입니다...",
        
        # 结果页
        "result_title": "분석 결과", "confidence": "정확도",
        "points_earned": "획득 포인트",
        "disposal_guide": "🗑️ 배출 방법 가이드",
        "low_conf_msg": "⚠️ 확실하지 않습니다. 이물질이 많다면 일반쓰레기로 버려주세요.",
        "btn_scan_again": "다시 스캔하기", "btn_check_stats": "통계 확인",
        
        # 统计 & 个人
        "total_scans": "총 스캔", "eco_points": "에코 포인트", "level": "레벨",
        "history_title": "최근 활동", "no_data": "아직 기록이 없습니다.",
        "badges_title": "🏆 나의 배지 컬렉션",
        "save": "저장", "username": "닉네임", "saved_msg": "저장되었습니다!",
        
        # 徽章名称
        "badge_starter": "시작하는 환경지킴이",
        "badge_bronze": "브론즈 리사이클러",
        "badge_silver": "실버 마스터",
        "badge_gold": "골드 레전드",
        "badge_locked": "잠김"
    },
    "zh": {
        "app_name": "EcoScan AI",
        "tagline": "AI 识别 + 韩国分类标准",
        "nav_home": "🏠 首页", "nav_scan": "📸 扫描", "nav_insights": "📊 统计", "nav_profile": "👤 我的",
        
        "hero_title": "垃圾分类不再头疼",
        "hero_subtitle": "拍照识别，获取正确的韩国垃圾分类指南",
        
        "step1_title": "1. 拍照上传", "step1_desc": "上传垃圾照片",
        "step2_title": "2. AI 识别", "step2_desc": "获取分类建议",
        "step3_title": "3. 赚取积分", "step3_desc": "积累环保贡献",
        
        "guide_plastic": "塑料", "guide_plastic_desc": "清洗并去标签",
        "guide_vinyl": "塑料包装", "guide_vinyl_desc": "必须干净",
        "guide_paper": "纸张", "guide_paper_desc": "压扁处理",
        "guide_trash": "一般垃圾", "guide_trash_desc": "脏污物品",
        "quick_guide_title": "📋 快速指南",

        "upload_btn": "📂 上传照片", "camera_btn": "📷 拍照",
        "scan_action": "🔍 开始识别",
        "analyzing": "AI 正在分析...",
        
        "result_title": "识别结果", "confidence": "置信度",
        "points_earned": "获得积分",
        "disposal_guide": "🗑️ 韩国处理指南",
        "low_conf_msg": "⚠️ 看起来有点模糊或混合，建议作为一般垃圾处理。",
        "btn_scan_again": "继续扫描", "btn_check_stats": "查看统计",
        
        "total_scans": "总次数", "eco_points": "积分", "level": "等级",
        "history_title": "最近记录", "no_data": "暂无数据",
        "badges_title": "🏆 成就徽章",
        "save": "保存", "username": "昵称", "saved_msg": "保存成功!",
        
        "badge_starter": "环保新手",
        "badge_bronze": "铜牌达人",
        "badge_silver": "银牌大师",
        "badge_gold": "金牌传奇",
        "badge_locked": "未解锁"
    },
    "en": {
        "app_name": "EcoScan AI",
        "tagline": "Smart Recycling Assistant",
        "nav_home": "🏠 Home", "nav_scan": "📸 Scan", "nav_insights": "📊 Stats", "nav_profile": "👤 Profile",
        
        "hero_title": "Sort Waste Correctly",
        "hero_subtitle": "Snap a photo to get AI sorting guide",
        
        "step1_title": "1. Capture", "step1_desc": "Take a photo",
        "step2_title": "2. Analyze", "step2_desc": "Get sorting rules",
        "step3_title": "3. Reward", "step3_desc": "Earn Eco Points",
        
        "guide_plastic": "Plastic", "guide_plastic_desc": "Wash & Label Off",
        "guide_vinyl": "Vinyl", "guide_vinyl_desc": "Must be Clean",
        "guide_paper": "Paper", "guide_paper_desc": "Flatten it",
        "guide_trash": "General", "guide_trash_desc": "Dirty Items",
        "quick_guide_title": "📋 Quick Guide",

        "upload_btn": "📂 Upload", "camera_btn": "📷 Camera",
        "scan_action": "🔍 Identify",
        "analyzing": "Analyzing...",
        
        "result_title": "Result", "confidence": "Confidence",
        "points_earned": "Points",
        "disposal_guide": "🗑️ Disposal Guide",
        "low_conf_msg": "⚠️ Unclear. If dirty/mixed, use General Trash.",
        "btn_scan_again": "Scan Again", "btn_check_stats": "Check Stats",
        
        "total_scans": "Scans", "eco_points": "Points", "level": "Level",
        "history_title": "Recent History", "no_data": "No data yet",
        "badges_title": "🏆 Badges",
        "save": "Save", "username": "Username", "saved_msg": "Saved!",
        
        "badge_starter": "Eco Starter",
        "badge_bronze": "Bronze Sorter",
        "badge_silver": "Silver Master",
        "badge_gold": "Gold Legend",
        "badge_locked": "Locked"
    }
}

# ==================================================
# 5. 分类逻辑 (韩国标准) & 徽章配置
# ==================================================
CATEGORIES = {
    "plastic": {
        "name": {"zh": "硬塑料", "en": "Plastic", "kr": "플라스틱 (투명/용기)"},
        "icon": "🥤", "color": "#10b981", "points": 10,
        "prompts": ["clear plastic bottle", "hard plastic container", "shampoo bottle", "pet bottle"],
        "tips": {
            "zh": "清洗内部，撕掉标签，压扁。",
            "en": "Wash inside, remove label, compress.",
            "kr": "내용물을 비우고 헹군 후, 라벨을 제거하고 압착하세요."
        }
    },
    "vinyl": {
        "name": {"zh": "塑料包装(Vinyl)", "en": "Vinyl/Wrapper", "kr": "비닐류 (라면/과자봉지)"},
        "icon": "🍬", "color": "#a855f7", "points": 5,
        "prompts": ["plastic snack bag", "ramen bag", "plastic wrapper", "crinkly plastic package"],
        "tips": {
            "zh": "必须干净！如果有油渍或食物残留，请丢一般垃圾。",
            "en": "Must be clean! If dirty, throw in General Trash.",
            "kr": "이물질이 없어야 합니다! 오염되었다면 일반쓰레기로 버리세요."
        }
    },
    "styrofoam": {
        "name": {"zh": "泡沫塑料", "en": "Styrofoam", "kr": "스티로폼"},
        "icon": "❄️", "color": "#94a3b8", "points": 7,
        "prompts": ["white styrofoam box", "clean styrofoam packaging"],
        "tips": {
            "zh": "仅限白色且干净的。去除胶带和运单。",
            "en": "White and clean only. Remove tape/labels.",
            "kr": "흰색의 깨끗한 것만 가능합니다. 테이프와 송장을 제거하세요."
        }
    },
    "paper": {
        "name": {"zh": "纸张/纸板", "en": "Paper/Box", "kr": "종이/박스"},
        "icon": "📦", "color": "#d97706", "points": 8,
        "prompts": ["cardboard box", "stack of newspapers", "paper document"],
        "tips": {
            "zh": "压平纸箱，去除胶带及订书钉。",
            "en": "Flatten boxes, remove tape and staples.",
            "kr": "박스는 납작하게 펴고, 테이프와 철심을 제거하세요."
        }
    },
    "can": {
        "name": {"zh": "金属罐", "en": "Metal Can", "kr": "캔류 (고철)"},
        "icon": "🥫", "color": "#3b82f6", "points": 15,
        "prompts": ["aluminum soda can", "tuna can", "metal food can"],
        "tips": {
            "zh": "清洗内部并压扁。",
            "en": "Wash inside and compress.",
            "kr": "내용물을 비우고 헹군 후, 가능한 찌그러뜨려주세요."
        }
    },
    "glass": {
        "name": {"zh": "玻璃瓶", "en": "Glass Bottle", "kr": "유리병"},
        "icon": "🍾", "color": "#0ea5e9", "points": 12,
        "prompts": ["glass bottle", "soju bottle", "beer bottle"],
        "tips": {
            "zh": "清洗干净。镜子和陶瓷不是玻璃回收物！",
            "en": "Wash clean. Mirrors/Ceramics are NOT recyclable.",
            "kr": "깨끗이 씻어주세요. 거울, 도자기는 재활용이 아닙니다!"
        }
    },
    "food": {
        "name": {"zh": "食物垃圾", "en": "Food Waste", "kr": "음식물 쓰레기"},
        "icon": "🍎", "color": "#facc15", "points": 2,
        "prompts": ["leftover food", "fruit peels", "vegetable scraps"],
        "tips": {
            "zh": "沥干水分。骨头、贝壳属于一般垃圾。",
            "en": "Drain water. Bones/Shells are General Trash.",
            "kr": "물기를 제거하세요. 뼈, 조개껍데기는 일반쓰레기입니다."
        }
    },
    "trash": {
        "name": {"zh": "一般垃圾", "en": "General Trash", "kr": "일반쓰레기 (종량제)"},
        "icon": "🗑️", "color": "#475569", "points": 1,
        "prompts": ["dirty tissue", "broken ceramic", "dirty food packaging", "mixed garbage"],
        "tips": {
            "zh": "使用计量垃圾袋。脏污无法清洗的物品都在这里。",
            "en": "Use standard trash bags. Dirty items go here.",
            "kr": "종량제 봉투를 사용하세요. 오염된 비닐/플라스틱은 여기입니다."
        }
    },
}

# 徽章配置 (积分阈值)
BADGES = [
    {"key": "badge_starter", "threshold": 0, "icon": "🌱", "color": "#10b981"},
    {"key": "badge_bronze", "threshold": 50, "icon": "🥉", "color": "#cd7f32"},
    {"key": "badge_silver", "threshold": 200, "icon": "🥈", "color": "#94a3b8"},
    {"key": "badge_gold", "threshold": 500, "icon": "🥇", "color": "#fbbf24"},
]

# ==================================================
# 6. AI 模型
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
        return None, None

processor, model = load_clip_model()

def classify_image(image):
    if not processor or not model: return "trash", 0.0
    
    # 图像预处理
    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    category_keys = list(CATEGORIES.keys())
    prompts = [f"a photo of {random.choice(CATEGORIES[key]['prompts'])}" for key in category_keys]
    
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = outputs.logits_per_image.softmax(dim=1)
    confidence, idx = torch.max(probs, dim=-1)
    
    category = category_keys[idx.item()]
    conf_val = confidence.item()
    
    if conf_val < 0.25: # 阈值
        return "trash", conf_val
        
    return category, conf_val

# ==================================================
# 7. UI 组件
# ==================================================
def render_navbar(t):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"<h3 style='margin:0;'>🌱 {t['app_name']}</h3><p style='margin:0;color:#64748b;font-size:0.9rem;'>{t['tagline']}</p>", unsafe_allow_html=True)
    with c2:
        # 语言选择器
        lang_map = {"kr": "🇰🇷 한국어", "en": "🇺🇸 English", "zh": "🇨🇳 中文"}
        new_lang = st.selectbox("Language", list(lang_map.keys()), format_func=lambda x: lang_map[x], index=list(lang_map.keys()).index(st.session_state.lang), label_visibility="collapsed")
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()
    st.markdown("---")

def render_badges_section(t):
    st.markdown(f"### {t['badges_title']}")
    cols = st.columns(4)
    current_points = st.session_state.total_points
    
    for idx, badge in enumerate(BADGES):
        is_unlocked = current_points >= badge['threshold']
        
        # 样式逻辑：解锁显示彩色，未解锁显示灰色+锁
        opacity = "1" if is_unlocked else "0.5"
        grayscale = "0" if is_unlocked else "100%"
        border_color = badge['color'] if is_unlocked else "#e2e8f0"
        
        # 徽章名称和状态文本
        badge_name = t[badge['key']]
        status_text = f"✅ {badge['threshold']} pts" if is_unlocked else f"🔒 {badge['threshold']} pts"
        
        with cols[idx]:
            st.markdown(f"""
            <div class="badge-card" style="border-color:{border_color}; opacity:{opacity}; filter:grayscale({grayscale});">
                <div style="font-size:3rem; margin-bottom:10px;">{badge['icon']}</div>
                <div style="font-weight:bold; font-size:0.9rem; margin-bottom:5px; height:40px; display:flex; align-items:center; justify-content:center;">{badge_name}</div>
                <div style="font-size:0.8rem; color:#64748b;">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)

# ==================================================
# 8. 主程序
# ==================================================
def main():
    t = TRANSLATIONS[st.session_state.lang]
    render_navbar(t)
    
    # Tabs
    tabs = [t['nav_home'], t['nav_scan'], t['nav_insights'], t['nav_profile']]
    if st.session_state.current_tab not in tabs:
        st.session_state.current_tab = tabs[0]
    
    selected_tab = st.radio("", tabs, horizontal=True, label_visibility="collapsed", key="current_tab")

    # --- 1. 首页 (HOME) ---
    if selected_tab == t['nav_home']:
        # Hero Banner
        st.markdown(f"""
        <div style='background:linear-gradient(135deg, #dcfce7, #bbf7d0); padding:40px 20px; border-radius:20px; text-align:center; margin-bottom:30px;'>
            <h1 style='color:#166534; font-size:2.2rem;'>{t['hero_title']}</h1>
            <p style='color:#15803d; font-size:1.1rem;'>{t['hero_subtitle']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 统计数据栏
        c1, c2, c3 = st.columns(3)
        c1.metric(t['total_scans'], len(st.session_state.history))
        c2.metric(t['eco_points'], st.session_state.total_points)
        c3.metric(t['level'], st.session_state.total_points // 100 + 1)
        
        # 步骤 (Steps)
        st.markdown("<br>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        steps = [
            ("📸", t['step1_title'], t['step1_desc']),
            ("🧠", t['step2_title'], t['step2_desc']),
            ("🎁", t['step3_title'], t['step3_desc'])
        ]
        for col, (icon, title, desc) in zip([sc1, sc2, sc3], steps):
            col.markdown(f"""
            <div style='text-align:center; padding:20px; background:#fff; border-radius:12px; border:1px solid #e2e8f0; height:100%;'>
                <div style='font-size:2rem; margin-bottom:10px;'>{icon}</div>
                <div style='font-weight:bold;'>{title}</div>
                <div style='font-size:0.8rem; color:#64748b;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 快速指南 (Quick Guide)
        st.markdown(f"### {t['quick_guide_title']}")
        gc1, gc2, gc3, gc4 = st.columns(4)
        guides = [
            ("🥤", t['guide_plastic'], t['guide_plastic_desc']),
            ("🍬", t['guide_vinyl'], t['guide_vinyl_desc']),
            ("📦", t['guide_paper'], t['guide_paper_desc']),
            ("🗑️", t['guide_trash'], t['guide_trash_desc'])
        ]
        for col, (icon, title, desc) in zip([gc1, gc2, gc3, gc4], guides):
            col.markdown(f"""
            <div style='text-align:center; padding:15px; background:#f8fafc; border-radius:10px;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='font-weight:bold; font-size:0.9rem;'>{title}</div>
                <div style='font-size:0.75rem; color:#64748b;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t['scan_action'], type="primary", use_container_width=True):
            st.session_state.current_tab = t['nav_scan']
            st.rerun()

    # --- 2. 扫描 (SCAN) ---
    elif selected_tab == t['nav_scan']:
        c1, c2 = st.columns(2)
        img_buffer = None
        with c1: 
            up = st.file_uploader(t['upload_btn'], type=["jpg","png","jpeg"], label_visibility="collapsed")
            if up: img_buffer = up
        with c2:
            cam = st.camera_input(t['camera_btn'], label_visibility="collapsed")
            if cam: img_buffer = cam
            
        if img_buffer:
            image = Image.open(img_buffer).convert("RGB")
            
            # --- 图片尺寸优化：使用三列布局，图片放中间 ---
            st.markdown("<br>", unsafe_allow_html=True)
            ic1, ic2, ic3 = st.columns([1, 2, 1]) 
            with ic2:
                st.image(image, use_container_width=True, caption="Preview")
            # ----------------------------------------
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(t['scan_action'], type="primary", use_container_width=True):
                with st.spinner(t['analyzing']):
                    time.sleep(0.8) # 模拟分析时间
                    cat, conf = classify_image(image)
                    info = CATEGORIES[cat]
                    
                    # 记录数据
                    pts = info['points']
                    st.session_state.total_points += pts
                    st.session_state.history.insert(0, {
                        "cat": cat, "conf": conf, "date": datetime.now().strftime("%m-%d %H:%M"), "pts": pts
                    })
                    
                    # 结果展示
                    st.balloons()
                    st.markdown(f"""
                    <div style='background-color:#fff; border:2px solid {info['color']}; border-radius:20px; padding:30px; text-align:center; box-shadow:0 10px 30px rgba(0,0,0,0.05); margin-top:20px;'>
                        <div style='font-size:5rem; margin-bottom:10px;'>{info['icon']}</div>
                        <h2 style='color:{info['color']}; margin:0;'>{info['name'][st.session_state.lang]}</h2>
                        <div style='font-size:1.5rem; font-weight:bold; color:{info['color']}; margin-top:10px;'>
                            +{pts} {t['eco_points']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 清洁指南
                    st.markdown(f"### {t['disposal_guide']}")
                    st.info(info['tips'][st.session_state.lang], icon="💡")
                    
                    if conf < 0.4:
                        st.warning(t['low_conf_msg'])
                        
                    ac1, ac2 = st.columns(2)
                    if ac1.button(t['btn_scan_again'], use_container_width=True):
                        st.rerun()
                    if ac2.button(t['btn_check_stats'], use_container_width=True):
                        st.session_state.current_tab = t['nav_insights']
                        st.rerun()

    # --- 3. 统计 (INSIGHTS) ---
    elif selected_tab == t['nav_insights']:
        if not st.session_state.history:
            st.info(t['no_data'])
        else:
            # 饼图
            counts = {}
            for h in st.session_state.history:
                counts[h['cat']] = counts.get(h['cat'], 0) + 1
            
            labels = [CATEGORIES[k]['name'][st.session_state.lang] for k in counts.keys()]
            values = list(counts.values())
            colors = [CATEGORIES[k]['color'] for k in counts.keys()]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6, marker=dict(colors=colors))])
            fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # 列表
            st.markdown(f"### {t['history_title']}")
            for h in st.session_state.history[:10]:
                info = CATEGORIES[h['cat']]
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; align-items:center; padding:12px; background:#fff; border-bottom:1px solid #f1f5f9;'>
                    <div style='display:flex; gap:10px; align-items:center;'>
                        <span style='font-size:1.5rem;'>{info['icon']}</span>
                        <div>
                            <div style='font-weight:bold;'>{info['name'][st.session_state.lang]}</div>
                            <div style='font-size:0.8rem; color:#94a3b8;'>{h['date']}</div>
                        </div>
                    </div>
                    <div style='color:{info['color']}; font-weight:bold;'>+{h['pts']}</div>
                </div>
                """, unsafe_allow_html=True)

    # --- 4. 个人 (PROFILE) ---
    elif selected_tab == t['nav_profile']:
        # 个人卡片
        st.markdown(f"""
        <div style='text-align:center; padding:40px; background:linear-gradient(to right, #6366f1, #8b5cf6); border-radius:20px; color:white; margin-bottom:30px;'>
            <div style='font-size:4rem; margin-bottom:10px;'>😎</div>
            <h2>{st.session_state.username}</h2>
            <p>Level {st.session_state.total_points // 100 + 1}</p>
            <div style='font-size:1.5rem; font-weight:bold; margin-top:10px;'>⭐ {st.session_state.total_points}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 徽章墙 (集成新功能)
        render_badges_section(t)
        
        st.markdown("---")
        
        # 设置
        new_name = st.text_input(t['username'], st.session_state.username)
        if new_name != st.session_state.username:
            if st.button(t['save'], type="primary"):
                st.session_state.username = new_name
                st.success(t['saved_msg'])
                time.sleep(1)
                st.rerun()

if __name__ == "__main__":
    main()
