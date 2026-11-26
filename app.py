import streamlit as st
import numpy as np
from scipy.io import wavfile
import io
import time

# 페이지 기본 설정
st.set_page_config(
    page_title="철도 백색소음 & 집중 타이머",
    page_icon="🚆",
    layout="centered"
)

# --- 오디오 합성 함수 (별도 mp3 파일 없이 코드로 소리 생성) ---
def generate_train_sound(duration=12, speed_kmh=100, rain_vol=0.2, rail_vol=0.7):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1. 기본 열차 주행 저주파 진동음 (브라운 노이즈 기반)
    white_noise = np.random.normal(0, 1, len(t))
    rumble = np.cumsum(white_noise)
    rumble = rumble / np.max(np.abs(rumble)) * 0.35

    # 2. 레일 이음매 소리 ("타-닥, 타-닥" 2축 대차 충격음 구현)
    rail_clicks = np.zeros_like(t)
    # 속도에 따른 이음매 통과 주기 계산 (표준 레일 길이 25m 기준)
    interval = 25.0 / (speed_kmh / 3.6)
    
    current_time = 0.2
    while current_time < duration - 0.5:
        # 앞바퀴 충격음 ("타-") & 뒷바퀴 충격음 ("-닥")
        for offset in [0.0, 0.08]:
            idx = int((current_time + offset) * sample_rate)
            click_len = int(0.04 * sample_rate)
            if idx + click_len < len(t):
                env = np.exp(-np.linspace(0, 12, click_len))
                click_sound = np.random.normal(0, 1, click_len) * env
                rail_clicks[idx:idx+click_len] += click_sound * 0.9
        current_time += interval

    # 3. 창밖 빗소리 (부드러운 필터링 노이즈)
    rain = np.convolve(np.random.normal(0, 1, len(t)), np.ones(15)/15, mode='same')
    rain = rain / np.max(np.abs(rain)) * 0.3

    # 사운드 믹싱
    mixed = (rumble * rail_vol) + (rail_clicks * rail_vol) + (rain * rain_vol)
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)  # 클리핑 방지 정규화
    audio_int16 = np.int16(mixed * 32767)

    # 메모리 버퍼에 WAV 형식으로 저장
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    return buf.getvalue()

# --- 노선 데이터 ---
ROUTES = {
    "🚄 KTX 경부선 (서울 ➔ 부산)": {"minutes": 165, "speed": 280, "desc": "시속 300km로 시원하게 내달리는 고속철도 주행음"},
    "🚆 ITX-새마을 경부선 (서울 ➔ 대전)": {"minutes": 115, "speed": 140, "desc": "적당한 속도감과 규칙적인 레일 이음매 소리"},
    "🚂 무궁화호 중앙선 (청량리 ➔ 안동)": {"minutes": 180, "speed": 100, "desc": "정겨운 '덜컹덜컹' 소리가 가장 잘 들리는 감성 노선"},
    "🚃 영동선 눈꽃열차 (강릉 ➔ 동해)": {"minutes": 45, "speed": 80, "desc": "가볍게 1교시 공부하기 좋은 45분 해안 노선"},
    "⏱️ 테스트용 미니 노선 (1분 시운전)": {"minutes": 1, "speed": 110, "desc": "타이머 작동 확인을 위한 짧은 시운전 코스"}
}

# --- UI 구성 ---
st.title("🚆 철도 백색소음 & 집중 타이머")
st.markdown("창밖 풍경을 떠올리며 목적지까지 열차와 함께 집중해 보세요.")

st.divider()

# 1. 노선 선택
selected_route = st.selectbox("탑승할 열차와 노선을 선택하세요:", list(ROUTES.keys()))
route_info = ROUTES[selected_route]

col1, col2 = st.columns(2)
with col1:
    st.metric(label="예상 소요 시간 (집중 목표)", value=f"{route_info['minutes']}분")
with col2:
    st.metric(label="평균 운행 속도", value=f"{route_info['speed']} km/h")

st.info(f"📢 **승무원 안내:** {route_info['desc']}")

# 2. 사운드 믹서 설정
st.subheader("🎛️ 열차 사운드 믹서")
c1, c2 = st.columns(2)
with c1:
    rail_vol = st.slider("레일 이음매 & 주행음 크기", 0.0, 1.0, 0.8)
with c2:
    rain_vol = st.slider("창밖 빗소리 크기", 0.0, 1.0, 0.3)

# 오디오 생성 및 플레이어 출력 (무한 반복 재생 지원)
audio_bytes = generate_train_sound(
    duration=12, 
    speed_kmh=route_info["speed"], 
    rain_vol=rain_vol, 
    rail_vol=rail_vol
)
st.audio(audio_bytes, format="audio/wav", loop=True)
st.caption("💡 재생 버튼(▶)을 누르면 목적지에 도착할 때까지 소리가 끊김 없이 반복 재생됩니다.")

st.divider()

# 3. 운행 타이머 (집중 모드)
st.subheader("⏱️ 열차 운행(집중) 타이머")

if st.button("🔔 출발 신호 울리기 (타이머 시작)", type="primary", use_container_width=True):
    total_seconds = route_info["minutes"] * 60
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.toast("삐-! 열차가 출발합니다. 집중을 시작하세요!")
    
    for elapsed in range(total_seconds + 1):
        remaining = total_seconds - elapsed
        mins, secs = divmod(remaining, 60)
        hours, mins = divmod(mins, 60)
        
        progress = elapsed / total_seconds
        progress_bar.progress(progress)
        
        status_text.markdown(
            f"### 🚉 목적지 도착까지 남은 시간: `{hours:02d}:{mins:02d}:{secs:02d}` ({int(progress*100)}% 운행 중)"
        )
        time.sleep(1)
        
    st.balloons()
    st.success("🎉 안내 말씀드립니다. 우리 열차는 잠시 후 최종 목적지에 도착합니다. 고생하셨습니다!")
