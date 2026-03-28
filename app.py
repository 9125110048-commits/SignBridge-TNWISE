import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from deep_translator import GoogleTranslator
import av

# --- RTC Configuration ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]}
)

# --- Finger Counting Logic ---
def count_fingers_logic(cnt):
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return 0
    defects = cv2.convexityDefects(cnt, hull_indices)
    if defects is None:
        return 0
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = np.linalg.norm(np.array(end) - np.array(far))
        b = np.linalg.norm(np.array(start) - np.array(far))
        c = np.linalg.norm(np.array(start) - np.array(end))
        angle = np.degrees(np.arccos(np.clip((b**2 + a**2 - c**2) / (2 * b * a + 1e-6), -1, 1)))
        if angle < 90 and d > 10000:
            count += 1
    return count + 1

# --- Webcam Processor ---
class VideoProcessor(VideoProcessorBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        roi = img[100:400, 100:400]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 20, 70], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 3000:
                finger_count = count_fingers_logic(cnt)
                cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                cv2.rectangle(img, (100, 100), (400, 400), (0, 200, 255), 2)
                cv2.putText(img, f"Fingers: {finger_count}", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.set_page_config(page_title="SignBridge AI", page_icon="🤟")
st.title("🤟 SignBridge: Two-Way Translation Platform")

choice = st.sidebar.radio("Select Mode", ("Sign to Text (Deaf User)", "Voice to Text (Hearing User)"))
lang_code = st.sidebar.selectbox(
    "Translate to",
    ["en", "ta", "hi", "es"],
    format_func=lambda x: {"en": "English", "ta": "Tamil", "hi": "Hindi", "es": "Spanish"}[x]
)

if choice == "Sign to Text (Deaf User)":
    st.subheader("Show your signs to the camera")
    webrtc_streamer(
        key="sign-to-text",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    st.write("Current translation will appear on the video feed.")

else:
    st.subheader("Translate Voice/Text for the Deaf User")
    st.info("Note: For the cloud demo, type the spoken message below to see the translation.")

    input_text = st.text_input("Hearing Person says:", "Hello, how are you?")

    if st.button("🔊 Translate for Deaf User"):
        try:
            translated = GoogleTranslator(source="auto", dest=lang_code).translate(input_text)
            st.success(f"Original: {input_text}")
            st.markdown(f"### 🤟 Translated to {lang_code}:")
            st.title(translated)
            st.caption("In a local deployment, this uses real-time Speech-to-Text.")
        except Exception as e:
            st.error(f"Translation failed: {e}")
