import streamlit as st
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import io
import time

st.set_page_config(
    page_title="철도 백색소음 & 집중 타이머",
    page_icon="🚆",
    layout="centered"
)

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def lowpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return lfilter(b, a, data)

def generate_metallic_train_sound(duration=15, speed_kmh=90, rain_vol=0.1, rail_vol=0.9):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    white = np.random.normal(0, 1, len(t))

    # 1. 차체 하부 베이스 진동 (쇳소리를 가리지 않도록 절제된 저음)
    rumble = lowpass_filter(white, cutoff=110, fs=sample_rate, order=3)
    rumble = (rumble / (np.max(np.abs(rumble)) + 1e-6)) * 0.22

    # 2. 강철 레일-차륜 연속 금속 마찰음 (쇠가 긁히는 "쉬이이-챠르르" 소리)
    steel_grind = bandpass_filter(white, lowcut=1800, highcut=4800, fs=sample_rate, order=3)
    # 레일 굴곡에 따라 쇠 긁히는 소리가 주기적으로 강해지는 변조
    grind_mod = 1.0 + 0.55 * np.sin(2 * np.pi * (speed_kmh / 28.0) * t)
    steel_grind = (steel_grind / (np.max(np.abs(steel_grind)) + 1e-6)) * 0.35 * grind_mod

    # 3. 강철 레일 이음매 타격음 ("깡-덕! ... 깡-덕!" 금속 공명음)
    rail_clicks = np.zeros_like(t)
    speed_ms = speed_kmh / 3.6
    interval = 25.0 / speed_ms
    axle_gap = 2.1 / speed_ms
    bogie_gap = 13.5 / speed_ms

    current_time = 0.2
    while current_time < duration - 1.0:
        hit_offsets = [0.0, axle_gap, bogie_gap, bogie_gap + axle_gap]
        for i, offset in enumerate(hit_offsets):
            idx = int((current_time + offset) * sample_rate)
            click_len = int(0.09 * sample_rate)
            if idx + click_len < len(t):
                ct = np.linspace(0, 0.09, click_len)
                # 앞바퀴("깡-")와 뒷바퀴("-덕")의 미세한 음정 차이
                pitch_shift = 1.08 if i % 2 == 0 else 0.94

                # 강철 레일이 울리는 비정수배 금속 배음 (쇠 종/레일 타격 공명)
                metal_ring = (
                    0.8 * np.sin(2 * np.pi * (155 * pitch_shift) * ct) * np.exp(-ct * 42) +
                    1.0 * np.sin(2 * np.pi * (435 * pitch_shift) * ct) * np.exp(-ct * 58) +
                    0.9 * np.sin(2 * np.pi * (975 * pitch_shift) * ct) * np.exp(-ct * 75) +
                    0.7 * np.sin(2 * np.pi * (1840 * pitch_shift) * ct) * np.exp(-ct * 95) +
                    0.5 * np.sin(2 * np.pi * (2890 * pitch_shift) * ct) * np.exp(-ct * 120)
                )

                # 이음매 틈새를 때리는 날카로운 고주파 쇠 파열음
                steel_snap = np.random.normal(0, 1, click_len) * np.exp(-ct * 90)
                steel_snap = bandpass_filter(steel_snap, lowcut=1200, highcut=5200, fs=sample_rate, order=2)

                accent = 1.0 if i % 2 == 0 else 0.85
                hit = (metal_ring * 1.15 + steel_snap * 1.4) * accent
                rail_clicks[idx:idx+click_len] += hit

        current_time += interval
        # 4. 창밖 빗소리
    rain_raw = np.random.normal(0, 1, len(t))
    rain = bandpass_filter(rain_raw, lowcut=800, highcut=4500, fs=sample_rate, order=2)
    rain = (rain / (np.max(np.abs(rain)) + 1e-6)) * 0.18

    # 전체 사운드 믹싱 (쇳소리 비율 극대화)
    train_total = rumble + steel_grind + (rail_clicks * 1.65)
    mixed = (train_total * rail_vol) + (rain * rain_vol)
    mixed = np.tanh(mixed * 1.2)
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)

    audio_int16 = np.int16(mixed * 32767)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    return buf.getvalue()

ROUTES = {
    "🚂 무궁화호 중앙선 (청량리 ➔ 안동)": {
        "minutes": 180,
        "speed": 90,
        "desc": "차가운 강철 레일 이음매 타격음('깡-덕')과 금속 마찰음이 선명한 노선"
    },
    "🚃 영동선 눈꽃열차 (강릉 ➔ 동해)": {
        "minutes": 45,
        "speed": 70,
        "desc": "느릿하게 선로를 긁으며 달리는 45분 집중 코스"
    },
    "🚆 ITX-새마을 경부선 (서울 ➔ 대전)": {
        "minutes": 115,
        "speed": 130,
        "desc": "경쾌한 속도감과 일정한 4박자 쇳소리가 돋보이는 노선"
    },
    "🚄 KTX 경부선 (서울 ➔ 부산)": {
        "minutes": 165,
        "speed": 250,
        "desc": "빠르게 연속되는 레일 마찰음과 고속 주행음"
    },
    "⏱️ 테스트용 미니 노선 (1분 시운전)": {
        "minutes": 1,
        "speed": 90,
        "desc": "타이머 작동 확인을 위한 짧은 시운전 코스"
    }
}

st.title("🚆 철도 백색소음 & 집중 타이머")
st.markdown("강철 레일 위를 달리는 쇠바퀴 소리를 들으며 목적지까지 집중해 보세요.")
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
    rail_vol = st.slider("강철 레일 & 쇠바퀴 소리 크기", 0.0, 1.0, 0.95)
with c2:
    rain_vol = st.slider("창밖 빗소리 크기", 0.0, 1.0, 0.1)

audio_bytes = generate_metallic_train_sound(
    duration=15,
    speed_kmh=route_info["speed"],
    rain_vol=rain_vol,
    rail_vol=rail_vol
)
st.audio(audio_bytes, format="audio/wav", loop=True)
st.caption("💡 빗소리를 0.0으로 내리면 쇠바퀴와 강철 레일이 부딪히는 소리만 순수하게 들립니다.")
st.divider()

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
