import streamlit as st
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import io
import os
import time

st.set_page_config(
    page_title="철도 백색소음 & 집중 타이머",
    page_icon="🚆",
    layout="centered"
)

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# 실제 MP3 파일을 불러와 속도 조절 및 빗소리를 믹싱하는 함수 (캐싱으로 빠른 로딩)
@st.cache_data(show_spinner="열차 주행음을 준비하고 있습니다...")
def load_and_mix_train_audio(mp3_path, speed_ratio=1.0, train_vol=0.9, rain_vol=0.15):
    # 1. 실제 기차 MP3 불러오기
    sound = AudioSegment.from_mp3(mp3_path)
    
    # 열차 노선(KTX, 무궁화 등)에 맞춰 재생 속도 자연스럽게 조절
    if speed_ratio != 1.0:
        new_frame_rate = int(sound.frame_rate * speed_ratio)
        sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
        sound = sound.set_frame_rate(44100)

    # 모노/스테레오 통일 및 numpy 배열 변환
    sound = sound.set_channels(1)
    sample_rate = sound.frame_rate
    samples = np.array(sound.get_array_of_samples(), dtype=np.float32)
    samples = samples / (np.max(np.abs(samples)) + 1e-6)

    # 루프 재생 시 앞뒤가 부드럽게 이어지도록 0.5초 크로스페이드 처리
    fade_len = int(0.5 * sample_rate)
    if len(samples) > fade_len * 2:
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        samples[:fade_len] *= fade_in
        samples[-fade_len:] *= fade_out

    # 2. 창밖 빗소리 생성 (실제 기차 소리와 어우러지는 부드러운 빗소리)
    rain_raw = np.random.normal(0, 1, len(samples))
    rain = bandpass_filter(rain_raw, lowcut=700, highcut=4200, fs=sample_rate, order=2)
    rain = (rain / (np.max(np.abs(rain)) + 1e-6)) * 0.25

    # 3. 실제 기차 소리 + 빗소리 믹싱
    mixed = (samples * train_vol) + (rain * rain_vol)
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)

    # WAV 메모리 버퍼로 변환
    audio_int16 = np.int16(mixed * 32767)
    out_seg = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    buf = io.BytesIO()
    out_seg.export(buf, format="wav")
    return buf.getvalue()
    ROUTES = {
    "🚂 무궁화호 중앙선 (청량리 ➔ 안동)": {
        "minutes": 180,
        "speed_ratio": 1.0,
        "desc": "원본 녹음 그대로의 리얼한 레일 이음매 타격음과 쇠바퀴 소리를 들을 수 있는 노선"
    },
    "🚃 영동선 눈꽃열차 (강릉 ➔ 동해)": {
        "minutes": 45,
        "speed_ratio": 0.88,
        "desc": "조금 더 느긋하고 묵직한 리듬으로 45분간 집중하기 좋은 코스"
    },
    "🚆 ITX-새마을 경부선 (서울 ➔ 대전)": {
        "minutes": 115,
        "speed_ratio": 1.15,
        "desc": "경쾌하게 속도를 높여 달리는 115분 집중 노선"
    },
    "🚄 KTX 경부선 (서울 ➔ 부산)": {
        "minutes": 165,
        "speed_ratio": 1.32,
        "desc": "빠른 템포의 레일 주행음과 함께하는 장거리 집중 코스"
    },
    "⏱️ 테스트용 미니 노선 (1분 시운전)": {
        "minutes": 1,
        "speed_ratio": 1.0,
        "desc": "타이머 작동 확인을 위한 짧은 시운전 코스"
    }
}

st.title("🚆 리얼 철도 백색소음 & 집중 타이머")
st.markdown("실제 열차 연결부(베스티ビュ)에서 녹음된 생생한 레일 소리와 함께 목적지까지 집중해 보세요.")
st.divider()

selected_route = st.selectbox("탑승할 열차와 노선을 선택하세요:", list(ROUTES.keys()))
route_info = ROUTES[selected_route]

col1, col2 = st.columns(2)
with col1:
    st.metric(label="예상 소요 시간 (집중 목표)", value=f"{route_info['minutes']}분")
with col2:
    st.metric(label="주행 템포 배속", value=f"{route_info['speed_ratio']}x")

st.info(f"📢 **승무원 안내:** {route_info['desc']}")

st.subheader("🎛️ 열차 사운드 믹서")
c1, c2 = st.columns(2)
with c1:
    train_vol = st.slider("실제 열차 & 레일 소리 크기", 0.0, 1.0, 0.95)
with c2:
    rain_vol = st.slider("창밖 빗소리 크기", 0.0, 1.0, 0.15)

MP3_FILE = "train.mp3"

if os.path.exists(MP3_FILE):
    audio_bytes = load_and_mix_train_audio(
        mp3_path=MP3_FILE,
        speed_ratio=route_info["speed_ratio"],
        train_vol=train_vol,
        rain_vol=rain_vol
    )
    st.audio(audio_bytes, format="audio/wav", loop=True)
    st.caption("💡 빗소리를 0.0으로 설정하면 순수 원본 기차 녹음만 들을 수 있습니다.")
else:
    st.error("⚠️ `train.mp3` 파일을 찾을 수 없습니다! GitHub 저장소의 `app.py`와 같은 폴더에 `train.mp3` 파일을 업로드해 주세요.")

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
