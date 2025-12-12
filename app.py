import streamlit as st
import time
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime, timedelta
import plotly.graph_objects as go
import torch
from transformers import CLIPProcessor, CLIPModel
import random

# ==================================================
# 1. 页面配置
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
    
    /* 导航栏 */
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
        padding: 10px 20px;
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
    
    /* 卡片与按钮 */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stImage {margin: 0;}
    
    /* 提示框样式 */
    .tip-box {
        background-color: #f0fdf4;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.95rem;
        color: #064e3b;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. Session State
# ==================================================
def init_session_state():
    defaults = {
        "history": [],
        "total_points": 0,
        "username": "EcoCitizen",
        "lang": "kr",  # 默认韩语
        "streak_days": 0,
        "last_scan_date": None,
        "total_co2_saved": 0,
        "achievements": [],
        "onboarding_done": False,
        "current_tab": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================================================
# 4. 韩国标准分类配置 (核心修改)
# ==================================================
# 逻辑：将垃圾分为 8 大类，并提供具体的清洁/处理指南
CATEGORIES = {
    "plastic": {
        "name": {"zh": "硬塑料 (Plastic)", "en": "Plastic Container", "kr": "플라스틱 (용기류)"},
        "icon": "🥤", "color": "#10b981", "points": 10, "co2_kg": 0.5,
        # 提示词强调刚性容器
        "prompts": ["clear plastic bottle", "hard plastic container", "shampoo bottle", "pet bottle"],
        "tips": {
            "zh": "1. 清空内容物并清洗。\n2. 撕掉标签贴纸。\n3. 瓶盖若材质不同请分开。",
            "en": "1. Empty and wash.\n2. Remove labels.\n3. Remove caps if different material.",
            "kr": "1. 내용물을 비우고 물로 헹궈주세요.\n2. 라벨(스티커)을 제거하세요.\n3. 뚜껑이 다른 재질이면 분리하세요."
        }
    },
    "vinyl": {
        "name": {"zh": "塑料包装 (Vinyl)", "en": "Vinyl/Wrapper", "kr": "비닐류 (필름/포장재)"},
        "icon": "🍬", "color": "#a855f7", "points": 5, "co2_kg": 0.4,
        # 提示词强调软包装
        "prompts": ["plastic snack bag", "ramen bag", "plastic wrapper", "crinkly plastic package", "vinyl bag"],
        "tips": {
            "zh": "1. 必须清洗干净，无食物残留。\n2. 如果脏了无法清洗，请作为'一般垃圾'丢弃。",
            "en": "1. Must be clean inside.\n2. If dirty, throw away as General Trash.",
            "kr": "1. 이물질이 없도록 깨끗이 씻어주세요.\n2. 음식물 등 오염이 심하면 '일반쓰레기'로 배출하세요."
        }
    },
    "paper": {
        "name": {"zh": "纸张/纸板", "en": "Paper/Box", "kr": "종이/박스"},
        "icon": "📦", "color": "#d97706", "points": 8, "co2_kg": 0.3,
        "prompts": ["cardboard box", "stack of newspapers", "milk carton", "paper document"],
        "tips": {
            "zh": "1. 压平纸箱。\n2. 去除胶带和订书钉。\n3. 纸包需要洗净晾干。",
            "en": "1. Flatten boxes.\n2. Remove tape and staples.\n3. Milk cartons must be washed.",
            "kr": "1. 박스는 납작하게 펼쳐주세요.\n2. 테이프와 철심을 제거하세요.\n3. 우유팩은 씻어서 말려주세요."
        }
    },
    "can": {
        "name": {"zh": "金属罐 (Can)", "en": "Metal Can", "kr": "캔류 (고철)"},
        "icon": "🥫", "color": "#3b82f6", "points": 15, "co2_kg": 0.8,
        "prompts": ["aluminum soda can", "tuna can", "metal food can", "compressed beer can"],
        "tips": {
            "zh": "1. 清洗内部。\n2. 尽量压扁。\n3. 放入金属回收箱。",
            "en": "1. Wash inside.\n2. Compress if possible.\n3. Place in metal bin.",
            "kr": "1. 내용물을 비우고 헹궈주세요.\n2. 가능한 압착(찌그러뜨려)해주세요.\n3. 뚜껑 등 다른 재질은 분리하세요."
        }
    },
    "glass": {
        "name": {"zh": "玻璃瓶", "en": "Glass Bottle", "kr": "유리병"},
        "icon": "🍾", "color": "#0ea5e9", "points": 12, "co2_kg": 0.6,
        "prompts": ["glass bottle", "soju bottle", "beer bottle", "glass jar"],
        "tips": {
            "zh": "1. 清洗干净。\n2. 瓶内不要放烟头。\n3. 镜子/陶瓷/碎玻璃属于特殊/一般垃圾！",
            "en": "1. Wash clean.\n2. No cigarette butts inside.\n3. Mirrors/Ceramics are NOT recyclable.",
            "kr": "1. 깨끗이 씻어주세요.\n2. 담배꽁초 등 이물질을 넣지 마세요.\n3. 거울, 깨진 유리, 도자기는 '불연성 쓰레기'입니다."
        }
    },
    "styrofoam": {
        "name": {"zh": "泡沫塑料", "en": "Styrofoam", "kr": "스티로폼"},
        "icon": "❄️", "color": "#94a3b8", "points": 7, "co2_kg": 0.2,
        "prompts": ["white styrofoam box", "clean styrofoam packaging"],
        "tips": {
            "zh": "1. 仅回收白色的、干净的。\n2. 去除所有胶带和标签。\n3. 彩色或脏的请按一般垃圾处理。",
            "en": "1. Only white and clean.\n2. Remove all tape/labels.\n3. Dirty ones go to Trash.",
            "kr": "1. 흰색의 깨끗한 것만 가능합니다.\n2. 테이프와 송장을 완벽히 제거하세요.\n3. 색이 있거나 오염된 것은 종량제 봉투에 버리세요."
        }
    },
    "food": {
        "name": {"zh": "食物垃圾", "en": "Food Waste", "kr": "음식물 쓰레기"},
        "icon": "🍎", "color": "#facc15", "points": 2, "co2_kg": 0.1,
        "prompts": ["leftover food", "fruit peels", "vegetable scraps", "banana peel", "food waste"],
        "tips": {
            "zh": "1. 沥干水分。\n2. 骨头、贝壳、硬核属于'一般垃圾'，不是食物垃圾！",
            "en": "1. Drain water.\n2. Bones/Shells/Seeds are GENERAL TRASH.",
            "kr": "1. 물기를 최대한 제거하세요.\n2. 뼈, 조개껍데기, 딱딱한 씨앗은 '일반쓰레기'입니다!"
        }
    },
    "electronics": {
        "name": {"zh": "特殊/电子垃圾", "en": "E-Waste/Special", "kr": "폐가전/특수"},
        "icon": "🔋", "color": "#ef4444", "points": 20, "co2_kg": 1.5,
        "prompts": ["used battery", "light bulb", "old mobile phone", "broken electronic device"],
        "tips": {
            "zh": "1. 不要丢入普通垃圾桶。\n2. 寻找专门的收集箱（如电池/灯泡回收盒）。",
            "en": "1. Do NOT put in standard bins.\n2. Find dedicated collection boxes.",
            "kr": "1. 일반 종량제 봉투에 버리지 마세요.\n2. 전용 수거함(폐건전지, 형광등)이나 주민센터에 배출하세요."
        }
    },
    "trash": {
        "name": {"zh": "一般垃圾", "en": "General Trash", "kr": "일반쓰레기 (종량제)"},
        "icon": "🗑️", "color": "#475569", "points": 1, "co2_kg": 0.0,
        # 包含脏污的、混合材质、无法回收的
        "prompts": ["dirty tissue", "broken ceramic", "dirty food packaging", "mixed garbage", "diaper", "pen", "toothbrush"],
        "tips": {
            "zh": "1. 使用标准垃圾袋 (Pay-as-you-go bag)。\n2. 任何脏污无法清洗的物品都属于此类。",
            "en": "1. Use standard trash bags.\n2. Dirty/Mixed items go here.",
            "kr": "1. 반드시 종량제 봉투(Pay-as-you-go bag)를 사용하세요.\n2. 씻어도 더러운 비닐/플라스틱은 여기에 버리세요."
        }
    },
}

# ==================================================
# 5. 多语言 UI 文本
# ==================================================
TRANSLATIONS = {
    "zh": {
        "app_name": "EcoScan 韩国版",
        "tagline": "AI 识别 + 韩国分类标准",
        "nav_home": "🏠 首页", "nav_scan": "📸 扫描", "nav_insights": "📊 记录", "nav_profile": "👤 我的",
        "hero_title": "在韩国，垃圾怎么扔？",
        "hero_subtitle": "拍照识别，获取正确的分类和清洁指南",
        "upload_btn": "📤 上传照片", "camera_btn": "📷 拍照",
        "scan_action": "🔍 立即识别",
        "analyzing": "AI 正在分析物体特征...",
        "result_match": "识别为", "confidence": "置信度",
        "disposal_guide": "🗑️ 韩国处理指南",
        "points_earned": "获得积分",
        "low_conf_msg": "🤔 看起来有点模糊，或者是混合垃圾。建议清洗后再次拍摄。",
        "total_scans": "总识别", "eco_points": "环保分", "level": "等级",
        "history_title": "最近记录", "no_data": "暂无记录",
        "save_success": "保存成功！", "username": "昵称",
        "btn_scan_again": "再扫一个", "btn_check_stats": "查看统计"
    },
    "en": {
        "app_name": "EcoScan KR",
        "tagline": "AI Sorting for Korea",
        "nav_home": "🏠 Home", "nav_scan": "📸 Scan", "nav_insights": "📊 Stats", "nav_profile": "👤 Profile",
        "hero_title": "Recycling in Korea?",
        "hero_subtitle": "Snap a photo to get sorting & cleaning rules",
        "upload_btn": "📤 Upload", "camera_btn": "📷 Camera",
        "scan_action": "🔍 Identify",
        "analyzing": "AI Analyzing...",
        "result_match": "Identified as", "confidence": "Confidence",
        "disposal_guide": "🗑️ Disposal Guide (Korea)",
        "points_earned": "Points",
        "low_conf_msg": "🤔 Looks unclear or mixed. Try cleaning it first.",
        "total_scans": "Scans", "eco_points": "Points", "level": "Level",
        "history_title": "Recent History", "no_data": "No data yet",
        "save_success": "Saved!", "username": "Username",
        "btn_scan_again": "Scan Again", "btn_check_stats": "Check Stats"
    },
    "kr": {
        "app_name": "에코스캔 AI",
        "tagline": "AI로 완벽한 분리수거",
        "nav_home": "🏠 홈", "nav_scan": "📸 스캔", "nav_insights": "📊 기록", "nav_profile": "👤 내 정보",
        "hero_title": "이 쓰레기, 어떻게 버리죠?",
        "hero_subtitle": "사진을 찍으면 올바른 분리배출 방법을 알려드립니다",
        "upload_btn": "📤 사진 업로드", "camera_btn": "📷 사진 촬영",
        "scan_action": "🔍 AI 분석 시작",
        "analyzing": "AI가 쓰레기 종류를 분석 중입니다...",
        "result_match": "분석 결과", "confidence": "정확도",
        "disposal_guide": "🗑️ 올바른 배출 방법",
        "points_earned": "획득 포인트",
        "low_conf_msg": "🤔 잘 모르겠습니다. 이물질이 묻어있다면 '일반쓰레기'일 확률이 높습니다.",
        "total_scans": "총 스캔", "eco_points": "에코 포인트", "level": "레벨",
        "history_title": "최근 활동", "no_data": "아직 기록이 없습니다.",
        "save_success": "저장되었습니다!", "username": "닉네임",
        "btn_scan_again": "계속 스캔하기", "btn_check_stats": "통계 보기"
    }
}

# ==================================================
# 6. AI 模型加载
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
        st.error(f"AI Model Error: {e}")
        return None, None

processor, model = load_clip_model()

# ==================================================
# 7. 核心逻辑
# ==================================================
def preprocess_image(image):
    # 增强对比度，帮助识别透明塑料和乙烯基
    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2) 
    return image

def classify_image(image):
    if not processor or not model: return "trash", 0.0
    
    processed_image = preprocess_image(image)
    
    # 构建 Prompt 列表
    category_keys = list(CATEGORIES.keys())
    prompts = []
    for key in category_keys:
        # 每个类别随机选一个 prompt 组合成句子
        p_text = f"a photo of {random.choice(CATEGORIES[key]['prompts'])}"
        prompts.append(p_text)
    
    inputs = processor(text=prompts, images=processed_image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = outputs.logits_per_image.softmax(dim=1)
    confidence, idx = torch.max(probs, dim=-1)
    
    category = category_keys[idx.item()]
    conf_val = confidence.item()
    
    # 稍微降低阈值，因为细分品类较多
    if conf_val < 0.28:
        return "trash", conf_val
        
    return category, conf_val

def get_level():
    return st.session_state.total_points // 100 + 1

# ==================================================
# 8. UI 组件
# ==================================================
def render_navbar(t):
    # 顶部状态栏
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"<h3 style='margin:0;text-align:left;'>🌱 {t['app_name']}</h3>", unsafe_allow_html=True)
        st.caption(t['tagline'])
    with c2:
        st.markdown(f"<div style='text-align:right;color:#10b981;font-weight:bold;'>⭐ {st.session_state.total_points} Pts</div>", unsafe_allow_html=True)
        
    # 语言选择
    lang_map = {"kr": "🇰🇷 한국어", "en": "🇺🇸 English", "zh": "🇨🇳 中文"}
    sel_lang = st.selectbox("Lang", list(lang_map.keys()), format_func=lambda x: lang_map[x], 
                           index=list(lang_map.keys()).index(st.session_state.lang), label_visibility="collapsed")
    if sel_lang != st.session_state.lang:
        st.session_state.lang = sel_lang
        st.rerun()

    st.markdown("---")

def render_scan_result(t, category, confidence, image):
    info = CATEGORIES[category]
    lang = st.session_state.lang
    
    st.balloons()
    
    # 结果卡片
    st.markdown(f"""
    <div style='background-color:#fff; border:2px solid {info['color']}; border-radius:20px; padding:30px; text-align:center; box-shadow:0 10px 30px rgba(0,0,0,0.1);'>
        <div style='font-size:5rem; margin-bottom:10px;'>{info['icon']}</div>
        <h2 style='color:{info['color']}; margin:0;'>{info['name'][lang]}</h2>
        <p style='color:#64748b; margin-top:5px;'>{t['confidence']}: {confidence:.0%}</p>
        <div style='font-size:1.5rem; font-weight:bold; color:{info['color']}; margin-top:15px;'>
            +{info['points']} {t['eco_points']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 核心：丢弃指南 (Disposal Tips)
    st.markdown(f"### {t['disposal_guide']}")
    st.info(info['tips'][lang], icon="💡")
    
    # 警告混合垃圾
    if confidence < 0.5:
        st.warning(t['low_conf_msg'])

    c1, c2 = st.columns(2)
    if c1.button(t['btn_scan_again'], use_container_width=True, type="primary"):
        st.rerun()
    if c2.button(t['btn_check_stats'], use_container_width=True):
        st.session_state.current_tab = t['nav_insights']
        st.rerun()

# ==================================================
# 9. 主程序
# ==================================================
def main():
    t = TRANSLATIONS[st.session_state.lang]
    render_navbar(t)
    
    # 导航 Tabs
    tabs = [t['nav_home'], t['nav_scan'], t['nav_insights'], t['nav_profile']]
    if st.session_state.current_tab not in tabs:
        st.session_state.current_tab = tabs[0]
        
    selected_tab = st.radio("", tabs, horizontal=True, label_visibility="collapsed", key="current_tab")

    # --- Home ---
    if selected_tab == t['nav_home']:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg, #dcfce7, #bbf7d0); padding:40px 20px; border-radius:20px; text-align:center; margin-bottom:30px;'>
            <h1 style='color:#166534; font-size:2.2rem;'>{t['hero_title']}</h1>
            <p style='color:#15803d; font-size:1.1rem;'>{t['hero_subtitle']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric(t['total_scans'], len(st.session_state.history))
        c2.metric(t['eco_points'], st.session_state.total_points)
        c3.metric(t['level'], get_level())
        
        st.markdown("### 📋 Quick Guide")
        guide_cols = st.columns(4)
        guides = [
            ("🥤", "Plastic", "Wash & Remove Label"),
            ("🍬", "Vinyl", "Must be Clean"),
            ("📦", "Paper", "Flatten & Remove Tape"),
            ("🗑️", "Trash", "Dirty Items Here")
        ]
        for col, (icon, title, desc) in zip(guide_cols, guides):
            col.markdown(f"<div style='text-align:center; background:#f8fafc; padding:15px; border-radius:10px; height:120px;'><div style='font-size:2rem;'>{icon}</div><strong>{title}</strong><br><span style='font-size:0.8rem; color:#64748b;'>{desc}</span></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t['scan_action'], type="primary", use_container_width=True):
            st.session_state.current_tab = t['nav_scan']
            st.rerun()

    # --- Scan ---
    elif selected_tab == t['nav_scan']:
        c1, c2 = st.columns(2)
        img_buffer = None
        with c1: 
            up = st.file_uploader(t['upload_btn'], type=["jpg","png","jpeg"])
            if up: img_buffer = up
        with c2:
            cam = st.camera_input(t['camera_btn'])
            if cam: img_buffer = cam
            
        if img_buffer:
            image = Image.open(img_buffer).convert("RGB")
            # 布局优化：图片居中且限制宽度
            cols = st.columns([1, 2, 1])
            with cols[1]:
                st.image(image, use_container_width=True, caption="Preview")
            
            if st.button(t['scan_action'], type="primary", use_container_width=True):
                with st.spinner(t['analyzing']):
                    time.sleep(0.8)
                    cat, conf = classify_image(image)
                    info = CATEGORIES[cat]
                    
                    # 保存记录
                    st.session_state.total_points += info['points']
                    st.session_state.history.insert(0, {
                        "cat": cat, "conf": conf, "date": datetime.now().strftime("%m-%d %H:%M"), "pts": info['points']
                    })
                    render_scan_result(t, cat, conf, image)

    # --- Insights ---
    elif selected_tab == t['nav_insights']:
        if not st.session_state.history:
            st.info(t['no_data'])
        else:
            # 数据统计
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

    # --- Profile ---
    elif selected_tab == t['nav_profile']:
        st.markdown(f"""
        <div style='text-align:center; padding:40px; background:linear-gradient(to right, #6366f1, #8b5cf6); border-radius:20px; color:white; margin-bottom:20px;'>
            <div style='font-size:4rem;'>😎</div>
            <h2>{st.session_state.username}</h2>
            <p>Level {get_level()} EcoCitizen</p>
        </div>
        """, unsafe_allow_html=True)
        
        new_name = st.text_input(t['username'], st.session_state.username)
        if new_name != st.session_state.username:
            if st.button(t['save_success']):
                st.session_state.username = new_name
                st.rerun()

if __name__ == "__main__":
    main()
