import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from string import ascii_lowercase
from googletrans import Translator
from gtts import gTTS
import pygame
from PIL import Image
import time
import speech_recognition as sr
from auth import init_db, register_user, login_user, reset_password

# ── Init DB ───────────────────────────────────────────────────────────────────
init_db()

# ================== CONFIG ==================
IMG_SIZE    = 50
DATADIR     = 'dataset'
AUDIO_FOLDER = 'test'
MODEL_PATH  = 'CNN.model'

LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)}

if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# ================== UTILS ==================
def alphabet_position(text):
    text = text.lower()
    return [LETTERS[char] for char in text if char in LETTERS]

# ================== MAIN APP ==================
def main_app():
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(MODEL_PATH)

    model      = load_model()
    CATEGORIES = os.listdir(DATADIR)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user['username']}")
        st.caption(st.session_state.user['email'])
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    st.markdown("<h1 style='color:#3498db;'>🧠 AI Sign & Speech Translator</h1>",
                unsafe_allow_html=True)

    option = st.radio(
        "Choose Mode:",
        ['🖐️ Sign Language to Text (Deaf Mode)', '🗣️ Speech to Sign (Dumb Mode)']
    )

    # ── Sign to Text ──────────────────────────────────────────────────────────
    if option == '🖐️ Sign Language to Text (Deaf Mode)':
        if st.button("Start Camera"):
            stframe = st.empty()
            cap     = cv2.VideoCapture(0)
            box_size = 300
            det = [
                'hi how are you', 'i dont know', 'what is your name', 'who are you',
                'what is this', 'where are you', 'how are you', 'i am hungry',
                'i am ironman', 'i love you', 'i hate you', 'i am sick',
                'i am sleeping', 'i am thirsty', 'i am in home', 'thankyou'
            ] * 2

            st.info("Press **Q** in the camera window to stop detection")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                hand    = frame[0:box_size, 0:box_size]
                gray    = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                arr     = resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

                prediction = model.predict(arr, verbose=0)
                index      = np.argmax(prediction)
                category   = CATEGORIES[index]

                cv2.rectangle(frame, (0, 0), (box_size, box_size), (0, 255, 0), 2)
                cv2.putText(frame, category, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')

                if "unknown" not in category:
                    time.sleep(0.3)
                    text       = det[index]
                    translated = Translator().translate(text, dest='ta').text
                    st.success(f"Detected: **{text}**")
                    st.write("Tamil:", translated)

                    audio_path = os.path.join(AUDIO_FOLDER, "output_audio.mp3")
                    gTTS(text=translated, lang='ta').save(audio_path)
                    try:
                        pygame.init()
                        pygame.mixer.init()
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        pygame.quit()
                    except Exception as e:
                        st.error(f"Audio error: {e}")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

    # ── Speech to Sign ────────────────────────────────────────────────────────
    else:
        if st.button("Start Recording"):
            r   = sr.Recognizer()
            mic = sr.Microphone()
            st.write("🎤 Speak now...")
            try:
                with mic as source:
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source, timeout=10, phrase_time_limit=8)

                recog = r.recognize_google(audio)
                st.success(f"You said: **{recog}**")

                images = []
                for char in recog:
                    if char == " ":
                        path = 'test/space.png'
                    elif char.lower() in LETTERS:
                        l    = alphabet_position(char)[0]
                        path = f'test/{int(l)}.jpg'
                    else:
                        continue
                    if os.path.exists(path):
                        images.append(Image.open(path).resize((150, 150)))

                if images:
                    st.image(images, width=75)
                else:
                    st.warning("No valid sign images found.")

            except sr.WaitTimeoutError:
                st.error("No speech detected. Try again.")
            except sr.UnknownValueError:
                st.error("Could not understand audio.")
            except sr.RequestError as e:
                st.error(f"Speech API error: {e}")


# ================== AUTH PAGES ==================
def login_page():
    st.markdown("### 🔐 Login")
    identifier = st.text_input("Username or Email", key="login_id")
    password   = st.text_input("Password", type="password", key="login_pw")

    if st.button("Login", use_container_width=True):
        if not identifier or not password:
            st.error("Please fill in all fields.")
        else:
            ok, msg, user = login_user(identifier, password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user      = user
                st.rerun()
            else:
                st.error(msg)

    with st.expander("Forgot Password?"):
        fp_email = st.text_input("Enter your email", key="fp_email")
        fp_new   = st.text_input("New password", type="password", key="fp_new")
        fp_conf  = st.text_input("Confirm new password", type="password", key="fp_conf")
        if st.button("Reset Password"):
            if fp_new != fp_conf:
                st.error("Passwords do not match.")
            elif len(fp_new) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                ok, msg = reset_password(fp_email, fp_new)
                st.success(msg) if ok else st.error(msg)


def register_page():
    st.markdown("### 📝 Register")
    username = st.text_input("Username", key="reg_user")
    email    = st.text_input("Email",    key="reg_email")
    password = st.text_input("Password", type="password", key="reg_pw")
    confirm  = st.text_input("Confirm Password", type="password", key="reg_conf")

    if st.button("Register", use_container_width=True):
        if not all([username, email, password, confirm]):
            st.error("Please fill in all fields.")
        elif password != confirm:
            st.error("Passwords do not match.")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            ok, msg = register_user(username, email, password)
            st.success(msg) if ok else st.error(msg)


def logout():
    st.session_state.logged_in = False
    st.session_state.user      = {}
    st.rerun()


# ================== ENTRY POINT ==================
st.set_page_config(
    page_title="Sign & Speech Translator",
    page_icon="🤟",
    layout="centered"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user      = {}

if st.session_state.logged_in:
    main_app()
else:
    st.markdown(
        "<h1 style='text-align:center;color:#3498db;'>🤟 Sign & Speech Translator</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:#888;'>Indian Sign Language · Text · Speech</p>",
        unsafe_allow_html=True
    )
    st.divider()
    tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])
    with tab_login:
        login_page()
    with tab_register:
        register_page()
