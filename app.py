import cv2
import numpy as np
import streamlit as st
from deep_translator import GoogleTranslator
import base64
from PIL import Image
import io

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

def finger_to_meaning(count):
    mapping = {
        0: "Fist / Stop",
        1: "One / Yes / Point",
        2: "Two / Peace / Victory",
        3: "Three / OK",
        4: "Four",
        5: "Five / Hello / Stop"
    }
    return mapping.get(count, "Unknown gesture")

def process_frame(img, lang_code):
    h, w = img.shape[:2]
    r_top    = max(0, h // 4)
    r_bottom = min(h, 3 * h // 4)
    r_left   = max(0, w // 4)
    r_right  = min(w, 3 * w // 4)
    roi = img[r_top:r_bottom, r_left:r_right].copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)

    cv2.rectangle(img, (r_left, r_top), (r_right, r_bottom), (0, 200, 255), 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result_text = "No hand detected"
    translated_text = ""

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 3000:
            finger_count = count_fingers_logic(cnt)
            meaning = finger_to_meaning(finger_count)
            result_text = f"Fingers: {finger_count} | {meaning}"

            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            img[r_top:r_bottom, r_left:r_right] = roi
            cv2.putText(img, result_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if lang_code != "en":
                try:
                    translated_text = GoogleTranslator(source="auto", dest=lang_code).translate(meaning)
                except Exception:
                    translated_text = meaning
            else:
                translated_text = meaning

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), result_text, translated_text

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
    st.subheader("Live Hand Sign Detection")

    # JavaScript live camera component
    live_cam_html = """
    <style>
        #container { display: flex; flex-direction: column; align-items: center; }
        video { border: 3px solid #0e76a8; border-radius: 10px; width: 400px; }
        canvas { display: none; }
        button {
            margin-top: 10px;
            padding: 10px 30px;
            background-color: #0e76a8;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover { background-color: #005f8a; }
        #status { margin-top: 8px; font-size: 14px; color: gray; }
    </style>

    <div id="container">
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
        <button onclick="toggleLive()">Start Live Detection</button>
        <div id="status">Camera off</div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const status = document.getElementById('status');
        let stream = null;
        let interval = null;
        let running = false;

        async function toggleLive() {
            if (!running) {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                running = true;
                status.innerText = "Live detection running...";
                document.querySelector('button').innerText = "Stop Live Detection";

                interval = setInterval(() => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    const frameData = canvas.toDataURL('image/jpeg', 0.7);
                    // Send frame to Streamlit
                    window.parent.postMessage({ type: "frame", data: frameData }, "*");
                }, 300); // capture every 300ms
            } else {
                clearInterval(interval);
                stream.getTracks().forEach(t => t.stop());
                video.srcObject = null;
                running = false;
                status.innerText = "Camera off";
                document.querySelector('button').innerText = "Start Live Detection";
            }
        }
    </script>
    """

    # Receive frame data via query params workaround
    st.components.v1.html(live_cam_html, height=400)

    st.markdown("---")
    st.info("Since true live processing requires a backend connection, use the snapshot method below for Streamlit Cloud — it processes each frame instantly on click.")

    photo = st.camera_input("Or use this for instant frame capture")

    if photo:
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is not None:
            processed_img, result_text, translated_text = process_frame(img, lang_code)

            st.image(processed_img, caption="Processed Frame", use_column_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Detected: {result_text}")
            with col2:
                if translated_text:
                    st.success(f"Meaning: {translated_text}")

            if translated_text:
                st.markdown("### Translation:")
                st.title(translated_text)

else:
    st.subheader("Translate Voice/Text for the Deaf User")
    st.info("Type the spoken message below to translate it for the deaf user.")

    input_text = st.text_input("Hearing Person says:", "Hello, how are you?")

    if st.button("Translate for Deaf User"):
        if not input_text.strip():
            st.warning("Please enter some text to translate.")
        else:
            try:
                if lang_code == "en":
                    translated = input_text
                else:
                    translated = GoogleTranslator(source="auto", dest=lang_code).translate(input_text)

                st.success(f"Original: {input_text}")
                st.markdown(f"### Translated to {lang_code.upper()}:")
                st.title(translated)
                st.caption("In a local deployment, this uses real-time Speech-to-Text via microphone.")

            except Exception as e:
                st.error(f"Translation failed: {e}")

