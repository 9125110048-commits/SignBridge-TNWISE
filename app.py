import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from googletrans import Translator
from gtts import gTTS
import speech_recognition as sr
import os

# --- Original Logic Integration ---
def count_fingers(cnt):
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3: return 0
    defects = cv2.convexityDefects(cnt, hull_indices)
    if defects is None: return 0
    
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start, end, far = tuple(cnt[s][0]), tuple(cnt[e][0]), tuple(cnt[f][0])
        a = np.linalg.norm(np.array(end) - np.array(far))
        b = np.linalg.norm(np.array(start) - np.array(far))
        c = np.linalg.norm(np.array(start) - np.array(end))
        angle = np.degrees(np.arccos(np.clip((b**2 + a**2 - c**2) / (2 * b * a + 1e-6), -1, 1)))
        if angle < 90 and d > 10000: finger_count += 1
    return min(finger_count + 1, 5)

# --- Web Interface ---
st.title("Inclusive Sign Translator & Voice Bridge")
mode = st.sidebar.selectbox("Select Mode", ["Sign to Speech (Multilingual)", "Speech to Sign (Vice Versa)"])
target_lang = st.sidebar.selectbox("Translation Language", ["en", "ta", "hi", "es", "fr"]) # Tamil, Hindi, etc.

translator = Translator()

if mode == "Sign to Speech (Multilingual)":
    st.subheader("Show your signs to the camera")
    
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # [Your existing skin mask and contour logic from sign translator 1.py goes here]
            # When a gesture is detected:
            # translated = translator.translate(gesture_text, dest=target_lang).text
            return img

    webrtc_streamer(key="sign-translate", video_transformer_factory=VideoProcessor)

elif mode == "Speech to Sign (Vice Versa)":
    st.subheader("Speak into the microphone")
    if st.button("Start Listening"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                st.write(f"You said: {text}")
                # Logic to display corresponding sign image
                # st.image(f"signs/{text.lower()}.png") 
            except:
                st.error("Could not understand audio")
