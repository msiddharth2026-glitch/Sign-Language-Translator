"""
Sign Language Two-Way Communication App
- Deaf Mode  : Webcam hand sign → speech (CNN model)
- Dumb Mode  : Microphone speech → Sign language images
"""

import os
import json
import hashlib
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from string import ascii_lowercase

import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
from gtts import gTTS
import pygame

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CNN.model")
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
AUDIO_DIR  = os.path.join(BASE_DIR, "audio")
TMP_AUDIO  = os.path.join(BASE_DIR, "audio_files", "output_audio.mp3")

# ── Model & categories ─────────────────────────────────────────────────────────
model      = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
CATEGORIES = sorted(os.listdir(DATA_DIR))
IMG_SIZE   = 50

# Phrase map (index → sentence) — matches training category order
PHRASES = [
    'hi how are you', 'i dont know', 'what is your name', 'who are you',
    'what is this', 'where are you', 'how are you', 'i am hungry',
    'i am ironman', 'i love you', 'i hate you', 'i am sick',
    'i am sleeping', 'i am thirsty', 'i am in home', 'thankyou',
    'hi how are you', 'i dont know', 'what is your name', 'who are you',
    'what is this', 'where are you', 'how are you', 'i am hungry',
    'i am ironman', 'i love you', 'i hate you', 'i am sick',
    'i am sleeping', 'i am thirsty', 'i am in home', 'thankyou',
]

LETTERS = {ch: str(i) for i, ch in enumerate(ascii_lowercase, start=1)}

# ── Helpers ────────────────────────────────────────────────────────────────────

def find_sign_image(index: int) -> str | None:
    """Return path to sign image for letter index (0-based), trying png/jpeg."""
    for ext in ("png", "jpeg", "jpg"):
        p = os.path.join(AUDIO_DIR, f"{index}.{ext}")
        if os.path.exists(p):
            return p
    return None


def find_space_image() -> str | None:
    for ext in ("png", "jpeg", "jpg"):
        p = os.path.join(AUDIO_DIR, f"space.{ext}")
        if os.path.exists(p):
            return p
    return None


def speak_tamil(text: str):
    """Speak text in English."""
    os.makedirs(os.path.dirname(TMP_AUDIO), exist_ok=True)
    tts = gTTS(text=text, lang="en")
    tts.save(TMP_AUDIO)
    pygame.mixer.init()
    pygame.mixer.music.load(TMP_AUDIO)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()


def resize_img(img, size=50):
    return cv2.resize(img, (size, size))


def build_grid(images, cols=10):
    """Stack images into a grid with `cols` columns."""
    blank = 255 * np.ones((50, 50, 3), dtype=np.uint8)
    rows, cur = [], []
    for img in images:
        cur.append(img)
        if len(cur) == cols:
            rows.append(cv2.hconcat(cur))
            cur = []
    if cur:
        cur += [blank] * (cols - len(cur))
        rows.append(cv2.hconcat(cur))
    return cv2.vconcat(rows)


# ── UI Drawing Helpers ─────────────────────────────────────────────────────────

# Color palette
C_BG       = (42, 23, 15)    # #0F172A in BGR
C_PANEL    = (59, 41, 30)    # #1E293B
C_ACCENT   = (248, 189, 56)  # #38BDF8 cyan
C_SUCCESS  = (78, 197, 34)   # #22C55E
C_WARNING  = (11, 158, 245)  # #F59E0B
C_TEXT     = (252, 250, 248) # #F8FAFC
C_TEXT2    = (184, 163, 148) # #94A3B8
C_BORDER   = (80, 60, 45)

FD = cv2.FONT_HERSHEY_DUPLEX
FC = cv2.FONT_HERSHEY_COMPLEX
FS = cv2.FONT_HERSHEY_SIMPLEX


def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """Draw a filled or outlined rounded rectangle."""
    if thickness == -1:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.rectangle(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.rectangle(img, (x2, y1+r), (x2, y2-r), color, thickness)
        for (cx, cy, a1, a2) in [(x1+r,y1+r,180,270),(x2-r,y1+r,270,360),
                                  (x1+r,y2-r,90,180),(x2-r,y2-r,0,90)]:
            cv2.ellipse(img, (cx,cy), (r,r), 0, a1, a2, color, thickness, cv2.LINE_AA)


def draw_glow(img, x1, y1, x2, y2, r, color, layers=3):
    """Draw a soft glow border around a rounded rect."""
    for i in range(layers, 0, -1):
        alpha = 0.15 * i / layers
        overlay = img.copy()
        draw_rounded_rect(overlay, x1-i*2, y1-i*2, x2+i*2, y2+i*2, r+i, color, 2)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def draw_progress_bar(img, x, y, w, h, pct, color, bg=(59,41,30), r=4):
    draw_rounded_rect(img, x, y, x+w, y+h, r, bg)
    if pct > 0:
        fill_w = max(r*2, int(w * pct))
        draw_rounded_rect(img, x, y, x+fill_w, y+h, r, color)


def put_text_aa(img, text, x, y, font, scale, color, thickness=1):
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_header(img, w, h):
    draw_rounded_rect(img, 0, 0, w, 56, 0, C_PANEL)
    cv2.line(img, (0, 56), (w, 56), C_BORDER, 1)
    # title
    put_text_aa(img, "AI Sign Language Translator", 20, 36, FD, 0.75, C_TEXT, 1)
    # mode label
    mode_txt = "Mode 1: Add Letters"
    (tw, _), _ = cv2.getTextSize(mode_txt, FS, 0.55, 1)
    put_text_aa(img, mode_txt, w//2 - tw//2, 36, FS, 0.55, C_ACCENT, 1)
    # shortcuts right
    shortcuts = "A=add  SPC=space  BKSP=del  ENTER=speak  Q=quit"
    (sw2, _), _ = cv2.getTextSize(shortcuts, FS, 0.42, 1)
    put_text_aa(img, shortcuts, w - sw2 - 16, 36, FS, 0.42, C_TEXT2, 1)


def draw_prediction_card(img, letter, conf, w, h):
    """Left panel: prediction card."""
    PAD = 16
    card_x1, card_y1 = PAD, 72
    card_x2, card_y2 = int(w * 0.38) - PAD, h - 100

    draw_rounded_rect(img, card_x1, card_y1, card_x2, card_y2, 12, C_PANEL)

    is_active = letter not in ("?", "—", "")
    accent = C_ACCENT if is_active else C_BORDER

    draw_glow(img, card_x1, card_y1, card_x2, card_y2, 12, accent)
    draw_rounded_rect(img, card_x1, card_y1, card_x2, card_y2, 12, accent, 2)

    cx = (card_x1 + card_x2) // 2
    mid_y = (card_y1 + card_y2) // 2

    # big letter
    disp = letter.upper() if is_active else "—"
    (lw, lh), _ = cv2.getTextSize(disp, FD, 4.0, 3)
    put_text_aa(img, disp, cx - lw//2, mid_y - 20, FD, 4.0, accent, 3)

    # status
    status = "Detecting..." if not is_active else "Ready"
    (sw2, _), _ = cv2.getTextSize(status, FS, 0.55, 1)
    put_text_aa(img, status, cx - sw2//2, mid_y + 40, FS, 0.55, C_TEXT2, 1)

    # confidence bar
    bar_y = card_y2 - 50
    bar_x = card_x1 + 20
    bar_w = (card_x2 - card_x1) - 40
    conf_col = C_SUCCESS if conf >= 0.85 else (C_WARNING if conf >= 0.6 else C_BORDER)
    draw_progress_bar(img, bar_x, bar_y, bar_w, 10, conf, conf_col)
    conf_txt = f"Confidence: {conf:.0%}"
    (ctw, _), _ = cv2.getTextSize(conf_txt, FS, 0.45, 1)
    put_text_aa(img, conf_txt, cx - ctw//2, bar_y - 8, FS, 0.45, C_TEXT2, 1)

    # label
    put_text_aa(img, "PREDICTION", card_x1+14, card_y1+22, FS, 0.45, C_TEXT2, 1)


def draw_camera_card(img, frame_roi, x1, y1, x2, y2, w, h):
    """Right panel: camera feed card — full frame shown, whole area is detection zone."""
    PAD = 16
    card_x1 = int(w * 0.38) + PAD
    card_y1 = 72
    card_x2 = w - PAD
    card_y2 = h - 100

    draw_rounded_rect(img, card_x1, card_y1, card_x2, card_y2, 12, C_PANEL)
    draw_rounded_rect(img, card_x1, card_y1, card_x2, card_y2, 12, C_BORDER, 2)

    feed_x1 = card_x1 + 12
    feed_y1 = card_y1 + 32
    feed_x2 = card_x2 - 12
    feed_y2 = card_y2 - 12
    feed_w  = feed_x2 - feed_x1
    feed_h  = feed_y2 - feed_y1

    if frame_roi is not None and feed_w > 0 and feed_h > 0:
        resized_feed = cv2.resize(frame_roi, (feed_w, feed_h))
        img[feed_y1:feed_y2, feed_x1:feed_x2] = resized_feed

    # border + corner accents on full feed area
    cv2.rectangle(img, (feed_x1, feed_y1), (feed_x2, feed_y2), C_ACCENT, 2)
    L = 18
    for (px, py, dx, dy) in [(feed_x1,feed_y1,1,1),(feed_x2,feed_y1,-1,1),
                              (feed_x1,feed_y2,1,-1),(feed_x2,feed_y2,-1,-1)]:
        cv2.line(img, (px, py), (px+dx*L, py), C_ACCENT, 3, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py+dy*L), C_ACCENT, 3, cv2.LINE_AA)

    put_text_aa(img, "Show your hand to the camera", feed_x1+8, feed_y1+22, FS, 0.5, C_ACCENT, 1)
    put_text_aa(img, "CAMERA FEED", card_x1+14, card_y1+22, FS, 0.45, C_TEXT2, 1)


def draw_bottom_panel(img, word_box, w, h):
    """Bottom status bar."""
    panel_y = h - 96
    draw_rounded_rect(img, 0, panel_y, w, h, 0, C_PANEL)
    cv2.line(img, (0, panel_y), (w, panel_y), C_BORDER, 1)

    word_str = "".join(word_box) if word_box else ""
    count    = len([c for c in word_box if c != " "])

    # word display
    put_text_aa(img, "WORD", 20, panel_y+22, FS, 0.45, C_TEXT2, 1)
    disp = word_str if word_str else "Start signing..."
    col  = C_TEXT if word_str else C_BORDER
    put_text_aa(img, disp, 20, panel_y+54, FD, 0.85, col, 1)

    # letter count badge
    badge_txt = f"{count} letters"
    (btw, _), _ = cv2.getTextSize(badge_txt, FS, 0.45, 1)
    draw_rounded_rect(img, w//2-btw//2-8, panel_y+10, w//2+btw//2+8, panel_y+30, 6, C_BORDER)
    put_text_aa(img, badge_txt, w//2-btw//2, panel_y+24, FS, 0.45, C_TEXT2, 1)

    # hints right
    hints = [("ENTER", "speak"), ("SPACE", "space"), ("BKSP", "delete"), ("A", "add")]
    hx = w - 20
    for key, action in hints:
        hint = f"{key}={action}"
        (hw, _), _ = cv2.getTextSize(hint, FS, 0.42, 1)
        hx -= hw + 20
        put_text_aa(img, key, hx, panel_y+22, FS, 0.42, C_ACCENT, 1)
        (kw, _), _ = cv2.getTextSize(key, FS, 0.42, 1)
        put_text_aa(img, f"={action}", hx+kw, panel_y+22, FS, 0.42, C_TEXT2, 1)


# ── Deaf Mode (Sign → Speech) ──────────────────────────────────────────────────

_deaf_running = False

def run_deaf(status_var: tk.StringVar, btn: tk.Button, root: tk.Tk):
    global _deaf_running
    _deaf_running = True
    btn.config(text="Stop Mode 1", command=lambda: stop_deaf(status_var, btn))
    status_var.set("Mode 1 — show sign, SPACE=add letter, ENTER=speak, BACKSPACE=delete, Q=quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open webcam.")
        _deaf_running = False
        btn.config(text="Start Mode 1", command=lambda: run_deaf(status_var, btn, root))
        return

    for _ in range(5):
        cap.read()

    WIN_NAME     = "Mode 1  |  SPACE=add  ENTER=speak  BACKSPACE=delete  Q=quit"
    FONT         = cv2.FONT_HERSHEY_SIMPLEX
    word_box     = []
    flash_frames = [0]
    live_letter  = [""]
    live_conf    = [0.0]
    frame_count  = [0]
    stable_letter = [""]
    stable_count  = [0]
    STABLE_NEEDED = 8   # frames the same letter must hold before showing

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WIN_NAME, 0, 0)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    def frame_loop():
        global _deaf_running
        if not _deaf_running:
            cap.release()
            cv2.destroyAllWindows()
            btn.config(text="Start Mode 1", command=lambda: run_deaf(status_var, btn, root))
            status_var.set("Mode 1 stopped.")
            return

        ret, frame = cap.read()
        if not ret:
            root.after(30, frame_loop)
            return

        h, w    = frame.shape[:2]

        # ── Build canvas ──────────────────────────────────────────────────────
        display = np.full((h, w, 3), C_BG, dtype=np.uint8)

        # Hand ROI — right side of frame, large box
        roi_w = int(w * 0.45)
        roi_h = int(h * 0.7)
        x1 = w - roi_w - 10
        y1 = int(h * 0.1)
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        # predict every 2nd frame — use full frame resized to 50x50
        frame_count[0] += 1
        if frame_count[0] % 2 == 0:
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            prepared = resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
            pred     = model(prepared)
            if isinstance(pred, dict):
                pred = list(pred.values())[0]
            pred = pred.numpy()
            live_conf[0] = float(np.max(pred))
            pred_class   = int(np.argmax(pred))
            pred_cat     = CATEGORIES[pred_class]
            raw_letter   = pred_cat if "unknown" not in pred_cat.lower() else "?"

            # stability lock — only update live_letter after STABLE_NEEDED consistent frames
            if raw_letter == stable_letter[0]:
                stable_count[0] += 1
            else:
                stable_letter[0] = raw_letter
                stable_count[0]  = 1

            if stable_count[0] >= STABLE_NEEDED:
                if live_conf[0] >= 0.90:
                    live_letter[0] = stable_letter[0]
                else:
                    live_letter[0] = "—"
            elif stable_count[0] == 1:
                # letter just changed — clear display until stable
                live_letter[0] = "—"

        # ── Flash ────────────────────────────────────────────────────────────
        if flash_frames[0] > 0:
            flash_frames[0] -= 1

        # ── Draw modern UI ────────────────────────────────────────────────────
        draw_header(display, w, h)
        draw_prediction_card(display, live_letter[0], live_conf[0], w, h)
        draw_camera_card(display, frame, x1, y1, x2, y2, w, h)
        draw_bottom_panel(display, word_box, w, h)

        cv2.imshow(WIN_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            # add current live letter to word box
            if live_letter[0] and live_letter[0] not in ("?", "—", ""):
                word_box.append(live_letter[0].upper())
                status_var.set(f"Added: {live_letter[0].upper()}  →  {''.join(word_box)}")
                flash_frames[0] = 6
        elif key == ord(" "):
            word_box.append(" ")
            status_var.set(f"Space added  →  {''.join(word_box)}")
        elif key == 13:  # ENTER
            if word_box:
                sentence = "".join(word_box)
                status_var.set(f"Speaking: {sentence}")
                threading.Thread(target=speak_tamil, args=(sentence,), daemon=True).start()
        elif key == 8:   # BACKSPACE
            if word_box:
                removed = word_box.pop()
                status_var.set(f"Removed: {removed}  →  {''.join(word_box)}")
        elif key == ord("c"):
            word_box.clear()
            status_var.set("Word cleared.")
        elif key == ord("q"):
            _deaf_running = False

        root.after(30, frame_loop)

    root.after(0, frame_loop)


def start_deaf(status_var, btn, root=None):
    if root:
        run_deaf(status_var, btn, root)


def stop_deaf(status_var, btn):
    global _deaf_running
    _deaf_running = False
    status_var.set("Stopping Mode 1…")


def start_deaf(status_var, btn):
    threading.Thread(target=run_deaf, args=(status_var, btn), daemon=True).start()

def stop_deaf(status_var, btn):
    global _deaf_running
    _deaf_running = False
    status_var.set("Stopping Deaf Mode…")


# ── Dumb Mode (Speech → Sign Images) ──────────────────────────────────────────

def show_sign_grid(text: str):
    """Display sign language images for each letter in text."""
    images, chars = [], []
    for ch in text.lower():
        if ch == " ":
            path = find_space_image(); chars.append(" ")
        elif ch in LETTERS:
            idx  = int(LETTERS[ch]) - 1
            path = find_sign_image(idx); chars.append(ch.upper())
        else:
            continue
        if path:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)

    if not images:
        messagebox.showwarning("No Images", "No sign images found for the text.")
        return

    CELL, PADDING, COLS = 160, 16, 6
    BG    = (18, 18, 30)
    BORDER = (0, 200, 255)
    FONT  = cv2.FONT_HERSHEY_SIMPLEX

    rows_needed = (len(images) + COLS - 1) // COLS
    WIN_W = COLS * (CELL + PADDING) + PADDING
    WIN_H = rows_needed * (CELL + PADDING + 28) + PADDING + 80

    canvas = np.full((WIN_H, WIN_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIN_W, 60), (10, 10, 22), -1)
    cv2.putText(canvas, f'"{text}"', (PADDING, 38), FONT, 0.85, (255, 220, 80), 2)
    cv2.putText(canvas, f"{len(images)} signs", (WIN_W-120, 38), FONT, 0.65, (120, 120, 180), 1)
    cv2.line(canvas, (0, 60), (WIN_W, 60), (40, 40, 80), 1)

    for i, (img, ch) in enumerate(zip(images, chars)):
        row = i // COLS; col = i % COLS
        x = PADDING + col * (CELL + PADDING)
        y = 70 + PADDING + row * (CELL + PADDING + 28)
        cv2.rectangle(canvas, (x-4, y-4), (x+CELL+4, y+CELL+4), (40, 40, 70), -1)
        cv2.rectangle(canvas, (x-4, y-4), (x+CELL+4, y+CELL+4), BORDER, 1)
        canvas[y:y+CELL, x:x+CELL] = cv2.resize(img, (CELL, CELL))
        label = ch if ch != " " else "SPC"
        lw = cv2.getTextSize(label, FONT, 0.6, 2)[0][0]
        cv2.putText(canvas, label, (x+(CELL-lw)//2, y+CELL+20), FONT, 0.6, (200, 200, 255), 2)

    WIN_NAME = f"Sign Language — {text}  (any key to close)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WIN_NAME, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_dumb(status_var: tk.StringVar, btn: tk.Button):
    """Mode 2 — show a dialog to choose Speech or Text input."""
    # ── Sub-mode chooser dialog ───────────────────────────────────────────────
    dialog = tk.Toplevel()
    dialog.title("Mode 2 — Choose Input")
    dialog.configure(bg="#0d0d28")
    dialog.resizable(False, False)
    dialog.grab_set()
    sw = dialog.winfo_screenwidth(); sh = dialog.winfo_screenheight()
    W, H = 420, 320
    dialog.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

    tk.Label(dialog, text="Mode 2: Text to Sign", font=("Segoe UI", 14, "bold"),
             bg="#0d0d28", fg="#38bdf8").pack(pady=(24, 4))
    tk.Label(dialog, text="Choose how to input your text",
             font=("Segoe UI", 9), bg="#0d0d28", fg="#94a3b8").pack()

    tk.Frame(dialog, bg="#1e293b", height=1).pack(fill="x", padx=30, pady=16)

    def do_speech():
        dialog.destroy()
        btn.config(state="disabled")
        status_var.set("Mode 2 (Speech): listening… speak now")
        def _run():
            r   = sr.Recognizer()
            mic = sr.Microphone()
            try:
                with mic as source:
                    r.adjust_for_ambient_noise(source, duration=1)
                    audio = r.listen(source, timeout=10, phrase_time_limit=8)
                text = r.recognize_google(audio, language="en-US")
                status_var.set(f"Recognised: \"{text}\"")
                show_sign_grid(text)
            except sr.WaitTimeoutError:
                status_var.set("No speech detected. Try again.")
            except sr.UnknownValueError:
                status_var.set("Could not understand audio. Try again.")
            except sr.RequestError as e:
                status_var.set(f"Speech API error: {e}")
            finally:
                btn.config(state="normal")
        threading.Thread(target=_run, daemon=True).start()

    def do_text():
        dialog.destroy()
        # ── Text input dialog ─────────────────────────────────────────────────
        text_win = tk.Toplevel()
        text_win.title("Mode 2 — Type Text")
        text_win.configure(bg="#0d0d28")
        text_win.resizable(False, False)
        text_win.grab_set()
        TW, TH = 480, 200
        text_win.geometry(f"{TW}x{TH}+{(sw-TW)//2}+{(sh-TH)//2}")

        tk.Label(text_win, text="Enter text to convert to sign language:",
                 font=("Segoe UI", 10), bg="#0d0d28", fg="#94a3b8").pack(pady=(20, 6))

        ent = tk.Entry(text_win, font=("Segoe UI", 13), bg="#0a0a20", fg="white",
                       insertbackground="white", relief="flat",
                       highlightthickness=1, highlightbackground="#313244",
                       highlightcolor="#38bdf8")
        ent.pack(padx=30, fill="x", ipady=8)
        ent.focus_set()

        msg = tk.Label(text_win, text="", font=("Segoe UI", 9),
                       bg="#0d0d28", fg="#f38ba8")
        msg.pack(pady=4)

        def submit():
            t = ent.get().strip()
            if not t:
                msg.config(text="Please enter some text.")
                return
            text_win.destroy()
            status_var.set(f"Showing signs for: \"{t}\"")
            show_sign_grid(t)
            btn.config(state="normal")

        tk.Button(text_win, text="Show Signs", command=submit,
                  bg="#1e3a5f", fg="white", activebackground="#1a4f82",
                  font=("Segoe UI", 11, "bold"), relief="flat",
                  padx=20, pady=8, cursor="hand2", bd=0).pack(pady=12)
        text_win.bind("<Return>", lambda e: submit())

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_frame = tk.Frame(dialog, bg="#0d0d28")
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="🎙  Speech Input", command=do_speech,
              bg="#1e3a5f", fg="white", activebackground="#1a4f82",
              font=("Segoe UI", 11, "bold"), relief="flat",
              padx=24, pady=14, cursor="hand2", bd=0,
              width=14).grid(row=0, column=0, padx=16)

    tk.Button(btn_frame, text="⌨  Text Input", command=do_text,
              bg="#1a3d2b", fg="white", activebackground="#1f5238",
              font=("Segoe UI", 11, "bold"), relief="flat",
              padx=24, pady=14, cursor="hand2", bd=0,
              width=14).grid(row=0, column=1, padx=16)

    tk.Button(dialog, text="Cancel", command=dialog.destroy,
              bg="#0d0d28", fg="#94a3b8", activebackground="#1e293b",
              font=("Segoe UI", 9), relief="flat", cursor="hand2").pack(pady=(8, 0))


def start_dumb(status_var, btn):
    threading.Thread(target=run_dumb, args=(status_var, btn), daemon=True).start()


# ── GUI ────────────────────────────────────────────────────────────────────────

# ── Auth helpers ──────────────────────────────────────────────────────────────
USERS_FILE = os.path.join(BASE_DIR, "users.json")

def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def auth_register(username: str, password: str) -> tuple:
    users = _load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = _hash(password)
    _save_users(users)
    return True, "Account created! You can now log in."

def auth_login(username: str, password: str) -> tuple:
    users = _load_users()
    if username not in users:
        return False, "Username not found."
    if users[username] != _hash(password):
        return False, "Incorrect password."
    return True, "Login successful."



# ── Single unified window ──────────────────────────────────────────────────────
def run_app():
    import random, math

    root = tk.Tk()
    root.title("Sign Language to Text/Speech Translation")
    root.configure(bg="#050510")
    root.overrideredirect(True)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{sw}x{sh}+0+0")
    root.lift()

    canvas = tk.Canvas(root, bg="#050510", highlightthickness=0)
    canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
    cx, cy = sw // 2, sh // 2

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — SPLASH
    # ══════════════════════════════════════════════════════════════════════════
    def phase_splash(on_done):
        running = [True]

        stars = []
        for _ in range(120):
            x = random.randint(0, sw); y = random.randint(0, sh)
            r = random.uniform(0.8, 2.5)
            col = random.choice(["#1a1a3a","#2a2a5a","#3a3a7a","#4a4a8a"])
            s = canvas.create_oval(x-r, y-r, x+r, y+r, fill=col, outline="", tags="splash")
            stars.append((s, x, y, random.uniform(0.3, 1.2)))

        def twinkle(i=0):
            if not running[0]: return
            for s, x, y, speed in stars:
                t = (i * speed) % 60
                bright = int(40 + 30 * abs(t - 30) / 30)
                canvas.itemconfig(s, fill=f"#{bright:02x}{bright:02x}{min(bright+60,255):02x}")
            root.after(80, lambda: twinkle(i+1))

        R = 110
        rings, ring_colors = [], ["#89b4fa","#74c7ec","#cba6f7","#f38ba8","#a6e3a1"]
        for i, col in enumerate(ring_colors):
            rs = R + i * 18
            ring = canvas.create_oval(cx-rs, cy-rs-60, cx+rs, cy+rs-60,
                                      outline=col, width=1, dash=(4,8), tags="splash")
            rings.append((ring, rs, col))

        def spin_rings(angle=0):
            if not running[0]: return
            for idx, (ring, rs, col) in enumerate(rings):
                canvas.itemconfig(ring, dashoffset=int(angle*(0.5+idx*0.3))%12,
                                  outline=ring_colors[(idx+int(angle/30))%len(ring_colors)])
            root.after(30, lambda: spin_rings(angle+1))

        logo_bg = canvas.create_oval(cx-R, cy-R-60, cx+R, cy+R-60,
                                     fill="#0f0f2a", outline="#89b4fa", width=3, tags="splash")
        canvas.create_text(cx, cy-60, text="🤟", font=("Segoe UI Emoji",72),
                           fill="white", tags="splash")
        for gi, gcol in enumerate(["#0d0d2a","#111130","#151540"]):
            gr = R + 6 + gi*10
            canvas.create_oval(cx-gr, cy-gr-60, cx+gr, cy+gr-60,
                               outline=gcol, width=8, tags="splash")

        title_id = canvas.create_text(cx, cy+100, text="", font=("Segoe UI",32,"bold"),
                                      fill="#89b4fa", anchor="center", tags="splash")
        sub_id   = canvas.create_text(cx, cy+148, text="", font=("Segoe UI",16),
                                      fill="#6c7086", anchor="center", tags="splash")
        tag_id   = canvas.create_text(cx, cy+182, text="", font=("Segoe UI",11,"italic"),
                                      fill="#313244", anchor="center", tags="splash")

        BAR_W = int(sw*0.4); bx, by = cx-BAR_W//2, cy+220
        canvas.create_rectangle(bx, by, bx+BAR_W, by+6, fill="#1e1e2e", outline="", tags="splash")
        bar_rect  = canvas.create_rectangle(bx, by, bx, by+6, fill="#89b4fa", outline="", tags="splash")
        status_id = canvas.create_text(cx, by+22, text="Initialising…",
                                       font=("Segoe UI",10), fill="#45475a", anchor="center", tags="splash")

        TOTAL_MS = 3800
        msgs = ["Initialising…","Loading model…","Preparing camera…","Setting up audio…","Almost ready…","✓  Ready!"]

        def type_text(item, text, idx=0, delay=45):
            if not running[0]: return
            canvas.itemconfig(item, text=text[:idx])
            if idx <= len(text):
                root.after(delay, lambda: type_text(item, text, idx+1, delay))

        def animate_bar(step=0, total=80):
            if not running[0]: return
            pct = step/total
            canvas.coords(bar_rect, bx, by, bx+int(BAR_W*pct), by+6)
            r_v=int(137*(1-pct)+166*pct); g_v=int(180*(1-pct)+227*pct); b_v=int(250*(1-pct)+161*pct)
            canvas.itemconfig(bar_rect, fill=f"#{r_v:02x}{g_v:02x}{b_v:02x}")
            canvas.itemconfig(status_id, text=msgs[min(int(pct*len(msgs)),len(msgs)-1)])
            if step < total:
                root.after(TOTAL_MS//total, lambda: animate_bar(step+1, total))

        def pulse_logo(i=0):
            if not running[0]: return
            new_r = int(R*(1.0+0.04*abs((i%20)-10)/10))
            canvas.coords(logo_bg, cx-new_r, cy-new_r-60, cx+new_r, cy+new_r-60)
            root.after(40, lambda: pulse_logo(i+1))

        def finish_splash():
            running[0] = False
            root.after(80, lambda: [canvas.delete("splash"), on_done()])

        root.after(0,    twinkle)
        root.after(0,    spin_rings)
        root.after(0,    pulse_logo)
        root.after(300,  lambda: type_text(title_id, "Sign Language", delay=55))
        root.after(1400, lambda: type_text(sub_id, "to Text / Speech Translation", delay=38))
        root.after(2000, lambda: type_text(tag_id, "Bridging communication, one sign at a time.", delay=28))
        root.after(400,  animate_bar)
        root.after(TOTAL_MS+200, finish_splash)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — LOGIN
    # ══════════════════════════════════════════════════════════════════════════
    def phase_login(on_done):
        canvas.configure(bg="#0a0a18")
        root.configure(bg="#0a0a18")

        W, H = 440, 480
        lx, ly = (sw-W)//2, (sh-H)//2

        canvas.create_rectangle(lx, ly, lx+W, ly+H, fill="#0d0d28",
                                 outline="#89b4fa", width=2, tags="login")
        canvas.create_text(lx+W//2, ly+52, text="🤟",
                           font=("Segoe UI Emoji",36), fill="white", tags="login")
        canvas.create_text(lx+W//2, ly+100, text="Sign Language App",
                           font=("Segoe UI",18,"bold"), fill="#89b4fa", tags="login")
        canvas.create_text(lx+W//2, ly+124, text="Login or create an account to continue",
                           font=("Segoe UI",9), fill="#4a5a7a", tags="login")

        lbl_cfg = {"bg":"#0d0d28","fg":"#6c7086","font":("Segoe UI",9)}
        ent_cfg = {"bg":"#0a0a20","fg":"white","insertbackground":"white",
                   "relief":"flat","font":("Segoe UI",11),
                   "highlightthickness":1,"highlightbackground":"#313244","highlightcolor":"#89b4fa"}

        user_lbl   = tk.Label(root, text="Username", **lbl_cfg)
        user_ent   = tk.Entry(root, **ent_cfg)
        pw_lbl     = tk.Label(root, text="Password", **lbl_cfg)
        pw_ent     = tk.Entry(root, show="●", **ent_cfg)
        pw2_lbl    = tk.Label(root, text="Confirm Password", **lbl_cfg)
        pw2_ent    = tk.Entry(root, show="●", **ent_cfg)
        msg_lbl    = tk.Label(root, text="", bg="#0d0d28", font=("Segoe UI",9), wraplength=320)
        action_btn = tk.Button(root, font=("Segoe UI",11,"bold"),
                               relief="flat", cursor="hand2", bd=0, pady=10)
        toggle_lbl = tk.Label(root, font=("Segoe UI",9), bg="#0d0d28", cursor="hand2")

        mode = ["login"]
        all_widgets = [user_lbl, user_ent, pw_lbl, pw_ent, pw2_lbl, pw2_ent,
                       msg_lbl, action_btn, toggle_lbl]

        def place_widgets():
            ox, oy = lx+60, ly
            user_lbl.place(x=ox, y=oy+148); user_ent.place(x=ox, y=oy+166, width=320, height=36)
            pw_lbl.place(x=ox, y=oy+214);   pw_ent.place(x=ox, y=oy+232, width=320, height=36)
            if mode[0] == "login":
                pw2_lbl.place_forget(); pw2_ent.place_forget()
                action_btn.config(text="Login", bg="#1e3a5f", fg="white",
                                  activebackground="#1a4f82", command=do_login)
                action_btn.place(x=ox, y=oy+288, width=320, height=44)
                toggle_lbl.config(text="Don't have an account? Register", fg="#4a5a7a")
                toggle_lbl.place(x=lx+W//2-110, y=oy+344)
            else:
                pw2_lbl.place(x=ox, y=oy+280); pw2_ent.place(x=ox, y=oy+298, width=320, height=36)
                action_btn.config(text="Create Account", bg="#1a3d2b", fg="white",
                                  activebackground="#1f5238", command=do_register)
                action_btn.place(x=ox, y=oy+352, width=320, height=44)
                toggle_lbl.config(text="Already have an account? Login", fg="#4a5a7a")
                toggle_lbl.place(x=lx+W//2-110, y=oy+408)
            msg_lbl.place(x=ox, y=ly+H-28, width=320)
            msg_lbl.config(text="")

        def show_msg(text, ok=True):
            msg_lbl.config(text=text, fg="#a6e3a1" if ok else "#f38ba8")

        def do_login():
            u, p = user_ent.get().strip(), pw_ent.get()
            if not u or not p: show_msg("Fill in all fields.", ok=False); return
            ok, msg = auth_login(u, p)
            if ok:
                for w in all_widgets: w.place_forget()
                canvas.delete("login")
                on_done()
            else:
                show_msg(msg, ok=False)

        def do_register():
            u, p, p2 = user_ent.get().strip(), pw_ent.get(), pw2_ent.get()
            if not u or not p or not p2: show_msg("Fill in all fields.", ok=False); return
            if p != p2: show_msg("Passwords do not match.", ok=False); return
            if len(p) < 4: show_msg("Min 4 characters.", ok=False); return
            ok, msg = auth_register(u, p)
            show_msg(msg, ok=ok)
            if ok: mode[0]="login"; place_widgets()

        toggle_lbl.bind("<Button-1>", lambda e: [
            mode.__setitem__(0, "register" if mode[0]=="login" else "login"),
            place_widgets()
        ])
        root.bind("<Return>", lambda e: do_login() if mode[0]=="login" else do_register())
        place_widgets()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — MAIN APP
    # ══════════════════════════════════════════════════════════════════════════
    def phase_main():
        canvas.delete("all")
        canvas.place_forget()
        root.configure(bg="#0f0f1a")
        root.overrideredirect(False)
        root.state("zoomed")

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TLabel",       background="#0f0f1a", foreground="#cdd6f4", font=("Segoe UI",12))
        style.configure("Title.TLabel", background="#0f0f1a", font=("Segoe UI",28,"bold"), foreground="#89b4fa")
        style.configure("Sub.TLabel",   background="#0f0f1a", font=("Segoe UI",11), foreground="#6c7086")
        style.configure("Status.TLabel",background="#090912", font=("Segoe UI",10), foreground="#a6e3a1")

        main_frame = tk.Frame(root, bg="#0f0f1a")
        main_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        header = tk.Frame(main_frame, bg="#0f0f1a")
        header.pack(fill="x", pady=(50,6))
        tk.Label(header, text="🤟", font=("Segoe UI Emoji",36), bg="#0f0f1a", fg="#cdd6f4").pack()
        ttk.Label(header, text="Sign Language to Text/Speech Translation", style="Title.TLabel").pack(pady=(6,4))
        ttk.Label(header, text="Mode 1  ·  hand sign → speech          Mode 2  ·  speech → sign images",
                  style="Sub.TLabel").pack()

        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", padx=60, pady=30)

        status_var = tk.StringVar(value="Ready — choose a mode to begin.")
        card_frame = tk.Frame(main_frame, bg="#0f0f1a")
        card_frame.pack(pady=10)

        def make_card(parent, emoji, title, subtitle, bg, hover, col, cmd_fn):
            card = tk.Frame(parent, bg=bg, padx=30, pady=24, cursor="hand2")
            card.grid(row=0, column=col, padx=30)
            tk.Label(card, text=emoji, font=("Segoe UI Emoji",30), bg=bg, fg="white").pack()
            tk.Label(card, text=title,    font=("Segoe UI",14,"bold"), bg=bg, fg="white").pack(pady=(6,2))
            tk.Label(card, text=subtitle, font=("Segoe UI",9), bg=bg, fg="#aaaacc").pack()
            btn = tk.Button(card, text=f"Start {title}", font=("Segoe UI",11,"bold"),
                            bg="white", fg=bg, activebackground="#eeeeee",
                            relief="flat", padx=20, pady=8, cursor="hand2", bd=0)
            btn.pack(pady=(16,0))
            btn.config(command=lambda: cmd_fn(status_var, btn))
            def on_enter(e): card.config(bg=hover); [w.config(bg=hover) for w in card.winfo_children() if isinstance(w,tk.Label)]
            def on_leave(e): card.config(bg=bg);    [w.config(bg=bg)    for w in card.winfo_children() if isinstance(w,tk.Label)]
            card.bind("<Enter>", on_enter); card.bind("<Leave>", on_leave)
            return btn

        make_card(card_frame, "👁", "Mode 1", "Show hand sign → speaks",
                  "#1e3a5f", "#1a4f82", 0, lambda sv, b: run_deaf(sv, b, root))

        # ── Mode 2 card with inline Speech + Text buttons ─────────────────────
        m2_bg, m2_hover = "#1a3d2b", "#1f5238"
        m2_card = tk.Frame(card_frame, bg=m2_bg, padx=30, pady=24)
        m2_card.grid(row=0, column=1, padx=30)

        tk.Label(m2_card, text="🎙", font=("Segoe UI Emoji",30), bg=m2_bg, fg="white").pack()
        tk.Label(m2_card, text="Mode 2", font=("Segoe UI",14,"bold"), bg=m2_bg, fg="white").pack(pady=(6,2))
        tk.Label(m2_card, text="Speech or text → sign images", font=("Segoe UI",9), bg=m2_bg, fg="#aaaacc").pack()

        btn_row = tk.Frame(m2_card, bg=m2_bg)
        btn_row.pack(pady=(16,0))

        def do_speech_mode(sv=None, b=None):
            sv = sv or status_var
            sv.set("Mode 2 (Speech): listening… speak now")
            def _run():
                r   = sr.Recognizer()
                mic = sr.Microphone()
                try:
                    with mic as source:
                        r.adjust_for_ambient_noise(source, duration=1)
                        audio = r.listen(source, timeout=10, phrase_time_limit=8)
                    text = r.recognize_google(audio, language="en-US")
                    sv.set(f"Recognised: \"{text}\"")
                    show_sign_grid(text)
                except sr.WaitTimeoutError:
                    sv.set("No speech detected. Try again.")
                except sr.UnknownValueError:
                    sv.set("Could not understand audio. Try again.")
                except sr.RequestError as e:
                    sv.set(f"Speech API error: {e}")
            threading.Thread(target=_run, daemon=True).start()

        def do_text_mode(sv=None, b=None):
            sv = sv or status_var
            text_win = tk.Toplevel()
            text_win.title("Mode 2 — Type Text")
            text_win.configure(bg="#0d0d28")
            text_win.resizable(False, False)
            text_win.grab_set()
            TW, TH = 480, 200
            text_win.geometry(f"{TW}x{TH}+{(root.winfo_x()+(root.winfo_width()-TW)//2)}+{(root.winfo_y()+(root.winfo_height()-TH)//2)}")

            tk.Label(text_win, text="Enter text to convert to sign language:",
                     font=("Segoe UI",10), bg="#0d0d28", fg="#94a3b8").pack(pady=(20,6))
            ent = tk.Entry(text_win, font=("Segoe UI",13), bg="#0a0a20", fg="white",
                           insertbackground="white", relief="flat",
                           highlightthickness=1, highlightbackground="#313244",
                           highlightcolor="#38bdf8")
            ent.pack(padx=30, fill="x", ipady=8)
            ent.focus_set()
            msg_l = tk.Label(text_win, text="", font=("Segoe UI",9), bg="#0d0d28", fg="#f38ba8")
            msg_l.pack(pady=4)

            def submit():
                t = ent.get().strip()
                if not t: msg_l.config(text="Please enter some text."); return
                text_win.destroy()
                sv.set(f"Showing signs for: \"{t}\"")
                show_sign_grid(t)

            tk.Button(text_win, text="Show Signs", command=submit,
                      bg="#1e3a5f", fg="white", activebackground="#1a4f82",
                      font=("Segoe UI",11,"bold"), relief="flat",
                      padx=20, pady=8, cursor="hand2", bd=0).pack(pady=12)
            text_win.bind("<Return>", lambda e: submit())

        tk.Button(btn_row, text="🎙 Speech", command=do_speech_mode,
                  bg="white", fg=m2_bg, activebackground="#eeeeee",
                  font=("Segoe UI",10,"bold"), relief="flat",
                  padx=14, pady=8, cursor="hand2", bd=0).grid(row=0, column=0, padx=6)

        tk.Button(btn_row, text="⌨ Text", command=do_text_mode,
                  bg="white", fg=m2_bg, activebackground="#eeeeee",
                  font=("Segoe UI",10,"bold"), relief="flat",
                  padx=14, pady=8, cursor="hand2", bd=0).grid(row=0, column=1, padx=6)

        def on_enter_m2(e): m2_card.config(bg=m2_hover); [w.config(bg=m2_hover) for w in m2_card.winfo_children() if isinstance(w,tk.Label)]
        def on_leave_m2(e): m2_card.config(bg=m2_bg);    [w.config(bg=m2_bg)    for w in m2_card.winfo_children() if isinstance(w,tk.Label)]
        m2_card.bind("<Enter>", on_enter_m2); m2_card.bind("<Leave>", on_leave_m2)

        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", padx=60, pady=30)
        tk.Button(main_frame, text="✕  Exit", command=root.destroy,
                  bg="#3b1f2b", fg="#f38ba8", activebackground="#4e2535",
                  font=("Segoe UI",10,"bold"), relief="flat",
                  padx=20, pady=8, cursor="hand2", bd=0).pack()

        status_frame = tk.Frame(root, bg="#090912", pady=8)
        status_frame.pack(fill="x", side="bottom")
        tk.Label(status_frame, text="●", font=("Segoe UI",10),
                 bg="#090912", fg="#a6e3a1").pack(side="left", padx=(14,4))
        ttk.Label(status_frame, textvariable=status_var, style="Status.TLabel").pack(side="left")

    # ── Chain phases ───────────────────────────────────────────────────────────
    phase_splash(lambda: phase_login(phase_main))
    root.mainloop()


if __name__ == "__main__":
    run_app()
