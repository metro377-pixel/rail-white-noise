import streamlit as st
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import io
import time

# 페이지 기본 설정
st.set_page_config(
    page_title="철도 백색소음 & 집중 타이머",
    page_icon="🚆",
    layout="centered"
)

# 저역 통과 필터 (날카로운 고음을 깎아 묵직한 저음 생성)
def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# 철제 기차 사운드 합성 함수
def generate_train_sound(duration=15, speed_kmh=100, rain_vol=0.2, rail_vol=0.8):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1. 차체 하부 및 선로 공명 저음
    white_noise = np.random.normal(0, 1, len(t))
    rumble = lowpass_filter(white_noise, cutoff=140, fs=sample_rate, order=3)
    lfo = 1.0 + 0.25 * np.sin(2 * np.pi * (speed_kmh / 60.0) * t)
    rumble = (rumble / (np.max(np.abs(rumble)) + 1e-6)) * 0.45 * lfo

    # 2. 레일 이음매 충격음 (4축 대차 리듬)
    rail_clicks = np.zeros_like(t)
    interval = 25.0 / (speed_kmh / 3.6)
    
    speed_ms = speed_kmh / 3.6
    axle_gap = 2.1 / speed_ms
    bogie_gap = 14.0 / speed_ms
    
    current_time = 0.3
    while current_time < duration - 1.0:
        hit_offsets = [0.0, axle_gap, bogie_gap, bogie_gap + axle_gap]
        
        for i, offset in enumerate(hit_offsets):
            idx = int((current_time + offset) * sample_rate)
            click_len = int(0.14 * sample_rate)
            
            if idx + click_len < len(t):
                ct = np.linspace(0, 0.14, click_len)
                base_freq = 78 if i % 2 == 0 else 64
                
                thud = np.sin(2 * np.pi * base_freq * ct) * np.exp(-ct * 28)
                thud += 0.5 * np.sin(2 * np.pi * (base_freq * 1.6) * ct) * np.exp(-ct * 35)
                
                metal_hit = np.random.normal(0, 1, click_len) * np.exp(-ct * 45)
                metal_hit = lowpass_filter(metal_hit, cutoff=650, fs=sample_rate, order=2)
                
                accent = 1.0 if i % 2 == 0 else 0.85
                combined_hit = (thud * 1.4 + metal_hit * 0.9) * accent
                rail_clicks[idx:idx+click_len] += combined_hit
                
        current_time += interval

    # 3. 창밖 빗소리
    rain_raw = np.random.normal(0, 1, len(t))
    rain = lowpass_filter(rain_raw, cutoff=1800, fs=sample_rate, order=2)
    rain = (rain / (np.max(np.abs(rain)) + 1e-6)) * 0.25

    # 사운드 믹싱
    mixed = (rumble * rail_vol) + (rail_clicks * rail_vol * 1.3) + (rain * rain_vol)
    mixed = np.tanh(mixed * 1.2)
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
    
    audio_int16 = np.int16(mixed * 32767)

    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    return buf.getvalue()

# 노선 데이터
ROUTES = {
    "🚂 무궁화호 중앙선 (청량리 ➔ 안동)": {
        "minutes": 180,
        "speed": 95,
        "desc": "묵직한 '쿵-덕, 쿵-덕' 레일 이음매 소리가 가장 잘 들리는 감성 노선"
    },
    "🚃 영동선 눈꽃열차 (강릉 ➔ 동해)": {
        "minutes": 45,
        "speed": 75,
        "desc": "느릿하게 흔들리며 가볍게 1교시 공부하기 좋은 45분 해안 노선"
    },
    "🚆 ITX-새마을 경부선 (서울 ➔ 대전)": {
        "minutes": 115,
        "speed": 135,
        "desc": "경쾌한 속도감과 규칙적인 4박자 리듬이 돋보이는 노선"
    },
    "🚄 KTX 경부선 (서울 ➔ 부산)": {
        "minutes": 165,
        "speed": 260,
        "desc": "빠르게 연속되는 레일 진동과 웅장한 고속 주행음"
    },
    "⏱️ 테스트용 미니 노선 (1분 시운전)": {
        "minutes": 1,
        "speed": 95,
        "desc": "타이머 작동 확인을 위한 짧은 시운전 코스"
    }
}

# UI 구성
st.title("🚆 철도 백색소음 & 집중 타이머")
st.markdown("창밖 풍경을 떠올리며 목적지까지 열차와 함께 집중해 보세요.")

st.divider()

selected_route = st.selectbox("탑승할 열차와 노선을 선택하세요:", list(ROUTES.keys()))
route_info = ROUTES[selected_route]

col1, col2 = st.columns(2)
with col1:
    st.metric(label="예상 소요 시간 (집중 목표)", value=f"{route_info['minutes']}분")
with col2:
    st.metric(label="평균 운행 속도", value=f"{route_info['speed']} km/h")

st.info(f"📢 **승무원 안내:** {route_info['desc']}")

st.subheader("🎛️ 열차 사운드 믹서")
c1, c2 = st.columns(2)
with c1:
    rail_vol = st.slider("레일 이음매 & 주행음 크기", 0.0, 1.0, 0.85)
with c2:
    rain_vol = st.slider("창밖 빗소리 크기", 0.0, 1.0, 0.25)

audio_bytes = generate_train_sound(
    duration=15, 
    speed_kmh=route_info["speed"], 
    rain_vol=rain_vol, 
    rail_vol=rail_vol
)
st.audio(audio_bytes, format="audio/wav", loop=True)
st.caption("💡 이어폰이나 스피커로 들으면 저음의 '쿵-덕' 울림이 훨씬 생생하게 들립니다.")

st.divider()

st.subheader("⏱️ 열차 운행(집중) 타이머")

if st.button("🔔 출발 신호 울
