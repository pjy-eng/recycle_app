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
# 4. 严格的多语言字典
# ==================================================
TRANSLATIONS = {
    "kr": {
        "app_name": "에코스캔 AI",
        "tagline": "AI로 완벽한 분리수거",
        "nav_home": "🏠 홈", "nav_scan": "📸 스캔", "nav_insights": "📊 통계", "nav_profile": "👤 내 정보",
        
        "hero_title": "이 쓰레기, 어떻게 버리죠?",
        "hero_subtitle": "사진을 찍으면 올바른 분리배출 방법을 알려드립니다",
        
        "step1_title": "1. 촬영/업로드", "step1_desc": "쓰레기 사진을 찍으세요",
        "step2_title": "2. AI 분석", "step2_desc": "종류와 배출 방법을 확인하세요",
        "step3_title": "3. 포인트 획득", "step3_desc": "환경을 지키고 보상을 받으세요",
        
        "guide_plastic": "플라스틱", "guide_plastic_desc": "헹구고 라벨 제거",
        "guide_vinyl": "비닐류", "guide_vinyl_desc": "깨끗한 상태로 배출",
        "guide_paper": "종이/박스", "guide_paper_desc": "펼쳐서 배출",
        "guide_trash": "일반쓰레기", "guide_trash_desc": "오염된 것은 여기로",
        "quick_guide_title": "📋 분리수거 핵심 가이드",

        "upload_btn": "📂 사진 업로드", "camera_btn": "📷 카메라",
        "scan_action": "🔍 분석 시작",
        "analyzing": "AI가 분석 중입니다...",
        
        "result_title": "분석 결과", "confidence": "정확도",
        "points_earned": "획득 포인트",
        "disposal_guide": "🗑️ 배출 방법 가이드",
        "low_conf_msg": "⚠️ 확실하지 않습니다. 이물질이 많다면 일반쓰레기로 버려주세요.",
        "btn_scan_again": "다시 스캔하기", "btn_check_stats": "통계 확인",
        
        "total_scans": "총 스캔", "eco_points": "에코 포인트", "level": "레벨",
        "history_title": "최근 활동", "no_data": "아직 기록이 없습니다.",
        "badges_title": "🏆 나의 배지 컬렉션",
        "save": "저장", "username": "닉네임", "saved_msg": "저장되었습니다!",
        
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
# 5. 分类逻辑 (严格韩国垃圾分类标准)
# ==================================================
CATEGORIES = {

    # ===============================
    # ♻️ 可回收 – 종이류
    # ===============================
    "paper": {
        "name": {"zh": "纸张/纸板", "en": "Paper", "kr": "종이류"},
        "icon": "📦",
        "color": "#d97706",
        "points": 8,
        "prompts": [
            "clean cardboard box flattened",
            "newspaper stack clean",
            "paper package without tape"
        ],
        "tips": {
            "zh": "压平后投放，去除胶带、订书钉。",
            "en": "Flatten and remove tape or staples.",
            "kr": "박스는 펼쳐서 테이프·철심 제거 후 배출하세요."
        }
    },

    # ===============================
    # ♻️ 可回收 – 플라스틱
    # ===============================
    "plastic": {
        "name": {"zh": "塑料容器", "en": "Plastic", "kr": "플라스틱"},
        "icon": "🥤",
        "color": "#10b981",
        "points": 10,
        "prompts": [
            "clean plastic bottle without label",
            "washed plastic container",
            "empty shampoo bottle clean"
        ],
        "tips": {
            "zh": "清洗干净，去除标签和异材质盖。",
            "en": "Wash clean, remove labels and caps.",
            "kr": "세척 후 라벨과 다른 재질의 뚜껑을 제거하세요."
        }
    },

    # ===============================
    # ♻️ 可回收 – 비닐류
    # ===============================
    "vinyl": {
        "name": {"zh": "塑料包装", "en": "Vinyl", "kr": "비닐류"},
        "icon": "🍬",
        "color": "#a855f7",
        "points": 5,
        "prompts": [
            "clean plastic wrapper",
            "dry snack bag no oil",
            "ramen plastic bag clean"
        ],
        "tips": {
            "zh": "必须无油污，否则算一般垃圾。",
            "en": "Only recyclable if clean and dry.",
            "kr": "기름기·이물질이 있으면 일반쓰레기입니다."
        }
    },

    # ===============================
    # ♻️ 可回收 – 캔류 (金属)
    # ===============================
    "can": {
        "name": {"zh": "金属罐", "en": "Metal Can", "kr": "캔류"},
        "icon": "🥫",
        "color": "#3b82f6",
        "points": 15,
        "prompts": [
            "empty aluminum soda can",
            "clean metal food can",
            "washed tuna can"
        ],
        "tips": {
            "zh": "清洗后压扁投放。",
            "en": "Rinse and compress before recycling.",
            "kr": "헹군 후 찌그러뜨려 배출하세요."
        }
    },

    # ===============================
    # ♻️ 可回收 – 유리병
    # ===============================
    "glass": {
        "name": {"zh": "玻璃瓶", "en": "Glass Bottle", "kr": "유리병"},
        "icon": "🍾",
        "color": "#0ea5e9",
        "points": 12,
        "prompts": [
            "clean glass bottle empty",
            "washed soju bottle",
            "beer bottle without cigarette"
        ],
        "tips": {
            "zh": "仅限瓶子，镜子、陶瓷不属于玻璃回收。",
            "en": "Bottles only. No mirrors or ceramics.",
            "kr": "병만 가능. 거울·도자기는 일반쓰레기입니다."
        }
    },

    # ===============================
    # ♻️ 可回收 – 스티로폼
    # ===============================
    "styrofoam": {
        "name": {"zh": "泡沫塑料", "en": "Styrofoam", "kr": "스티로폼"},
        "icon": "❄️",
        "color": "#94a3b8",
        "points": 7,
        "prompts": [
            "clean white styrofoam box",
            "white foam packaging clean"
        ],
        "tips": {
            "zh": "仅限白色且干净的。",
            "en": "Only white and clean styrofoam.",
            "kr": "흰색이고 깨끗한 것만 가능합니다."
        }
    },

    # ===============================
    # 🍎 食物垃圾 – 음식물
    # ===============================
    "food": {
        "name": {"zh": "厨余垃圾", "en": "Food Waste", "kr": "음식물쓰레기"},
        "icon": "🍎",
        "color": "#facc15",
        "points": 2,
        "prompts": [
            "food leftovers",
            "fruit peels",
            "vegetable scraps"
        ],
        "tips": {
            "zh": "去水，骨头、贝壳是一般垃圾。",
            "en": "Drain water. Bones and shells are trash.",
            "kr": "물기 제거. 뼈·조개껍데기는 일반쓰레기입니다."
        }
    },

    # ===============================
    # 🔋 特殊垃圾
    # ===============================
    "special": {
        "name": {"zh": "特殊垃圾", "en": "Special Waste", "kr": "특수쓰레기"},
        "icon": "🔋",
        "color": "#ef4444",
        "points": 0,
        "prompts": [
            "used battery",
            "fluorescent lamp",
            "medicine pills"
        ],
        "tips": {
            "zh": "送至指定回收点。",
            "en": "Dispose at special collection points.",
            "kr": "지정 수거함에 배출하세요."
        }
    },

    # ===============================
    # 🗑️ 一般垃圾 – 종량제
    # ===============================
    "trash": {
        "name": {"zh": "一般垃圾", "en": "General Trash", "kr": "일반쓰레기"},
        "icon": "🗑️",
        "color": "#475569",
        "points": 1,
        "prompts": [
            "dirty plastic packaging",
            "broken ceramic",
            "mirror glass",
            "greasy food wrapper",
            "mixed garbage"
        ],
        "tips": {
            "zh": "使用韩国指定垃圾袋。",
            "en": "Use official Korean trash bags.",
            "kr": "종량제 봉투를 사용하세요."
        }
    },
}

# ==================================================
# 6.AI 分类逻辑（韩国规则兜底）
# ==================================================
def classify_image(image):
    if not processor or not model:
        return "trash", 0.0

    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.2)

    category_keys = list(CATEGORIES.keys())
    prompts = [
        f"a photo of {random.choice(CATEGORIES[k]['prompts'])}"
        for k in category_keys
    ]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)
    confidence, idx = torch.max(probs, dim=1)

    conf_val = confidence.item()
    category = category_keys[idx.item()]

    # 🇰🇷 韩国规则：不确定 = 一般垃圾
    if conf_val < 0.28:
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
        opacity = "1" if is_unlocked else "0.5"
        grayscale = "0" if is_unlocked else "100%"
        border_color = badge['color'] if is_unlocked else "#e2e8f0"
        
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
        
        # 修复后的按钮逻辑：使用 on_click 回调
        def go_to_scan():
            st.session_state.current_tab = t['nav_scan']
            
        st.button(t['scan_action'], type="primary", use_container_width=True, on_click=go_to_scan)

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
            
            # --- 图片尺寸优化 ---
            st.markdown("<br>", unsafe_allow_html=True)
            ic1, ic2, ic3 = st.columns([1, 2, 1]) 
            with ic2:
                st.image(image, use_container_width=True, caption="Preview")
            # --------------------
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 按钮的回调不需要，因为这里不需要跳转页面，只需要执行逻辑
            if st.button(t['scan_action'], type="primary", use_container_width=True):
                with st.spinner(t['analyzing']):
                    time.sleep(0.8) 
                    cat, conf = classify_image(image)
                    info = CATEGORIES[cat]
                    
                    pts = info['points']
                    st.session_state.total_points += pts
                    st.session_state.history.insert(0, {
                        "cat": cat, "conf": conf, "date": datetime.now().strftime("%m-%d %H:%M"), "pts": pts
                    })
                    
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
                    
                    st.markdown(f"### {t['disposal_guide']}")
                    st.info(info['tips'][st.session_state.lang], icon="💡")
                    
                    if conf < 0.4:
                        st.warning(t['low_conf_msg'])
                        
                    ac1, ac2 = st.columns(2)
                    if ac1.button(t['btn_scan_again'], use_container_width=True):
                        st.rerun()
                    
                    # 使用回调跳转统计页
                    def go_to_insights():
                        st.session_state.current_tab = t['nav_insights']
                    ac2.button(t['btn_check_stats'], use_container_width=True, on_click=go_to_insights)

    # --- 3. 统计 (INSIGHTS) ---
    elif selected_tab == t['nav_insights']:
        if not st.session_state.history:
            st.info(t['no_data'])
        else:
            counts = {}
            for h in st.session_state.history:
                counts[h['cat']] = counts.get(h['cat'], 0) + 1
            
            labels = [CATEGORIES[k]['name'][st.session_state.lang] for k in counts.keys()]
            values = list(counts.values())
            colors = [CATEGORIES[k]['color'] for k in counts.keys()]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6, marker=dict(colors=colors))])
            fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
            
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
        st.markdown(f"""
        <div style='text-align:center; padding:40px; background:linear-gradient(to right, #6366f1, #8b5cf6); border-radius:20px; color:white; margin-bottom:30px;'>
            <div style='font-size:4rem; margin-bottom:10px;'>😎</div>
            <h2>{st.session_state.username}</h2>
            <p>Level {st.session_state.total_points // 100 + 1}</p>
            <div style='font-size:1.5rem; font-weight:bold; margin-top:10px;'>⭐ {st.session_state.total_points}</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_badges_section(t)
        
        st.markdown("---")
        
        new_name = st.text_input(t['username'], st.session_state.username)
        if new_name != st.session_state.username:
            if st.button(t['save'], type="primary"):
                st.session_state.username = new_name
                st.success(t['saved_msg'])
                time.sleep(1)
                st.rerun()

if __name__ == "__main__":
    main()
