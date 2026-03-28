import cv2
import numpy as np
import streamlit as st
from deep_translator import GoogleTranslator

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

# --- Finger count to meaning ---
def finger_to_meaning(count):
    mapping = {
        0: "✊ Fist / Stop",
        1: "☝️ One / Yes / Point",
        2: "✌️ Two / Peace / Victory",
        3: "🤟 Three / OK",
        4: "🖖 Four",
        5: "🖐️ Five / Hello / Stop"
    }
    return mapping.get(count, "Unknown gesture")

# --- Streamlit UI ---
st.set_page_config(page_title="SignBridge AI", page_icon="🤟")
st.title("🤟 SignBridge: Two-Way Translation Platform")

choice = st.sidebar.radio("Select Mode", ("Sign to Text (Deaf User)", "Voice to Text (Hearing User)"))
lang_code = st.sidebar.selectbox(
    "Translate to",
    ["en", "ta", "hi", "es"],
    format_func=lambda x: {"en": "English", "ta": "Tamil", "hi": "Hindi", "es": "Spanish"}[x]
)

# ─────────────────────────────────────────────
# MODE 1: Sign to Text
# ─────────────────────────────────────────────
if choice == "Sign to Text (Deaf User)":
    st.subheader("📷 Show your hand sign to the camera")
    st.info("💡 Tip: Use good lighting and keep your hand inside the frame for best results.")

    photo = st.camera_input("Take a photo of your hand sign")

    if photo:
        # Decode image
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Could not read the image. Please try again.")
        else:
            h, w = img.shape[:2]

            # Safe ROI — adapts to image size
            r_top    = max(0, h // 4)
            r_bottom = min(h, 3 * h // 4)
            r_left   = max(0, w // 4)
            r_right  = min(w, 3 * w // 4)
            roi = img[r_top:r_bottom, r_left:r_right]

            # Skin detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 20, 70], dtype="uint8")
            upper = np.array([20, 255, 255], dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)

            # Draw ROI box on image for display
            cv2.rectangle(img, (r_left, r_top), (r_right, r_bottom), (0, 200, 255), 2)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)

                if area > 3000:
                    finger_count = count_fingers_logic(cnt)
                    meaning = finger_to_meaning(finger_count)

                    # Draw contour on ROI
                    cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                    cv2.putText(img, f"Fingers: {finger_count}", (r_left, r_top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Show annotated image
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

                    # Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fingers Detected", finger_count)
                    with col2:
                        st.success(f"Gesture: {meaning}")

                    # Translate the meaning
                    if lang_code != "en":
                        try:
                            translated = GoogleTranslator(source="auto", dest=lang_code).translate(meaning)
                            st.markdown(f"### 🌐 Translated ({lang_code}):")
                            st.title(translated)
                        except Exception:
                            st.warning("Translation failed. Showing original.")
                    else:
                        st.markdown(f"### 🤟 Meaning:")
                        st.title(meaning)

                else:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.warning("⚠️ Hand detected but too small. Move your hand closer to the camera.")
            else:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.error("❌ No hand detected. Try better lighting or a plain background.")

# ─────────────────────────────────────────────
# MODE 2: Voice to Text
# ─────────────────────────────────────────────
else:
    st.subheader("🎙️ Translate Voice/Text for the Deaf User")
    st.info("Type the spoken message below to translate it for the deaf user.")

    input_text = st.text_input("Hearing Person says:", "Hello, how are you?")

    if st.button("🔊 Translate for Deaf User"):
        if not input_text.strip():
            st.warning("Please enter some text to translate.")
        else:
            try:
                if lang_code == "en":
                    translated = input_text
                else:
                    translated = GoogleTranslator(source="auto", dest=lang_code).translate(input_text)

                st.success(f"✅ Original: {input_text}")
                st.markdown(f"### 🌐 Translated to {lang_code.upper()}:")
                st.title(translated)
                st.caption("In a local deployment, this can use real-time Speech-to-Text via microphone.")

            except Exception as e:
                st.error(f"Translation failed: {e}")
```

---

**What changed from before:**

| Feature | Before | Now |
|---|---|---|
| Camera | `webrtc_streamer` (broken on cloud) | `st.camera_input()` ✅ |
| ROI | Fixed 100–400px box | Adapts to any image size ✅ |
| Gesture meaning | Only showed finger count | Maps count to real meaning ✅ |
| Translation | Always translated | Skips API call if already English ✅ |
| Error handling | Basic | Handles no hand, small hand, bad image ✅ |
| Display | No annotated image shown | Shows processed image with ROI box ✅ |

Also update your `requirements.txt` — remove the WebRTC lines:
```
opencv-python-headless
numpy
streamlit
deep-translator
