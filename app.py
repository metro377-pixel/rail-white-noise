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

# --- 필터 함수 (고주파 날카로움을 깎아 묵직한 저음 생성) ---
def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# --- 진짜 무거운 철제 기차 사운드 합성 함수 ---
def generate_train_sound(duration=15, speed_kmh=100, rain_vol=0.2, rail_vol=0.8):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1. 차체 하부 및 선로 공명 저음 (묵직하게 깔리는 "우우웅-" 소리)
    white_noise = np.random.normal(0, 1, len(t))
    rumble = lowpass_filter(white_noise, cutoff=140, fs=sample_rate, order=3)
    # 속도에 따라 미세하게 출렁이는 진동(LFO) 추가
    lfo = 1.0 + 0.25 * np.sin(2 * np.pi * (speed_kmh / 60.0) * t)
    rumble = (rumble / (np.max(np.abs(rumble)) + 1e-6)) * 0.45 * lfo

    # 2. 레일 이음매 충격음 ("쿵-덕 ... 쿵-덕" 4축 대차 리듬)
    rail_clicks = np.zeros_like(t)
    # 25m 표준 레일 기준 통과 주기
    interval = 25.0 / (speed_kmh / 3.6)
    
    # 앞대차(2축)와 뒷대차(2축)의 바퀴 간격 시간 계산
    speed_ms = speed_kmh / 3.6
    axle_gap = 2.1 / speed_ms       # 대차 내 앞뒤 바퀴 간격 (약 2.1m)
    bogie_gap = 14.0 / speed_ms     # 앞대차와 뒷대차 사이 거리 (약 14m)
    
    current_time = 0.3
    while current_time < duration - 1.0:
        # 한 량이 이음매를 지날 때 발생하는 4번의 바퀴 충격 시점
        hit_offsets = [0.0, axle_gap, bogie_gap, bogie_gap + axle_gap]
        
        for i, offset in enumerate(hit_offsets):
            idx = int((current_time + offset) * sample_rate)
            # 충격음이 길게 울리도록 0.14초 길이로 설정
            click_len = int(0.14 * sample_rate)
            
            if idx + click_len < len(t):
                ct = np.linspace(0, 0.14, click_len)
                # 바퀴마다 미세하게 다른 음높이 (앞바퀴는 높고 뒷바퀴는 낮게: "쿵-덕")
                base_freq = 78 if i % 2 == 0 else 64
                
                # 1) 묵직한 쇠바퀴 타격 저음 (65~80Hz 감쇠 사인파)
                thud = np.sin(2 * np.pi * base_freq * ct) * np.exp(-ct * 28)
                thud += 0.5 * np.sin(2 * np.pi * (base_freq * 1.6) * ct) * np.exp(-ct * 35)
                
                # 2) 레일 마찰 쇳소리 (저역 통과 필터를 거친 묵직한 타격 노이즈)
                metal_hit = np.random.normal(0, 1, click_len) * np.exp(-ct * 45)
                metal_hit = lowpass_filter(metal_hit, cutoff=650, fs=sample_rate, order=2)
                
                # 볼륨에 미세한 강약 변화 추가 (사람 귀에 자연스럽게 들리도록)
                accent = 1.0 if i % 2 == 0 else 0.85
                combined_hit = (thud * 1.4 + metal_hit * 0.9) * accent
                rail_clicks[idx:idx+click_len] += combined_hit
                
        current_time += interval

    # 3. 창밖 빗소리 (차창에 부딪히는 부드러운 빗방울 소리)
    rain_raw = np.random.normal(0, 1, len(t))
    rain = lowpass_filter(rain_raw, cutoff=180
