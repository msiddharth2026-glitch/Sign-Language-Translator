import tkinter as tk
import threading
import os
import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
from string import ascii_lowercase
from googletrans import Translator
from gtts import gTTS
from playsound import playsound

# Load trained model for hand sign recognition
model = tf.keras.models.load_model('CNN.model')
DATADIR = 'dataset'
CATEGORIES = os.listdir(DATADIR)  # Get all subfolder names as categories
IMG_SIZE = 50

LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)}

def alphabet_position(text):
    text = text.lower()
    numbers = [LETTERS[character] for character in text if character in LETTERS]
    return numbers  # Returns a list of positions as strings

def resize_image(image, width=50, height=50):
    return cv2.resize(image, (width, height))

def run_deaf():
    cap = cv2.VideoCapture(0)
    box_size = 300
    cv2.namedWindow('Hand Image')
    
    det = ['hi how are you','i dont know','what is your name','who are you','what is this','where are you','how are you','i am hungry','i am ironman','i love you','i hate you','i am sick','i am sleeping','i am thirsty','i am in home','thankyou','hi how are you','i dont know','what is your name','who are you','what is this','where are you','how are you','i am hungry','i am ironman','i love you','i hate you','i am sick','i am sleeping','i am thirsty','i am in home','thankyou']

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hand_box = gray_frame[0:box_size, 0:box_size]
        resized_frame = cv2.resize(hand_box, (IMG_SIZE, IMG_SIZE))
        prepared_image = np.array(resized_frame).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
        prediction = model.predict(prepared_image)
        predicted_class = np.argmax(prediction)
        predicted_category = CATEGORIES[predicted_class]
        
        if "unknown" not in predicted_category:
            detected_text = det[predicted_class]
            print("Detected Text:", detected_text)
            translator = Translator()
            translated_text = translator.translate(detected_text, dest='ta').text
            print("Translated Tamil Text:", translated_text)
            tts = gTTS(text=translated_text, lang='ta')
            tts.save("Tamil-Audio.mp3")
            playsound("Tamil-Audio.mp3")
            os.remove("Tamil-Audio.mp3")
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predicted_category, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (box_size, box_size), (0, 255, 0), 2)
        cv2.imshow('Hand Image', resized_frame)
        cv2.imshow('Hand Sign Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_dumb():
    r = sr.Recognizer()
    speech = sr.Microphone(device_index=1)
    with speech as source:
        print("Say something!")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    
    try:
        recog = r.recognize_google(audio, language='en-US')
        print("You said:", recog)
        images = []
        for char in recog:
            if char == " ":
                img_path = 'audio/space.jpg'
            elif char.lower() in LETTERS:
                l = alphabet_position(char)[0]
                k = int(l) - 1
                img_path = 'audio/' + str(k) + '.jpg'
            else:
                continue
            image = cv2.imread(img_path)
            if image is not None:
                images.append(resize_image(image))
        
        if images:
            images_per_row = 10
            row_images = []
            current_row = []
            for img in images:
                current_row.append(img)
                if len(current_row) == images_per_row:
                    row_images.append(cv2.hconcat(current_row))
                    current_row = []
            if current_row:
                while len(current_row) < images_per_row:
                    blank_image = resize_image(255 * np.ones_like(images[0]))
                    current_row.append(blank_image)
                row_images.append(cv2.hconcat(current_row))
            combined_image = cv2.vconcat(row_images)
            cv2.imshow('Detected Characters', combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No valid images found.")
    except sr.UnknownValueError:
        print("Speech not recognized")
    except sr.RequestError as e:
        print("Request error:", e)

def run_in_thread(func):
    thread = threading.Thread(target=func)
    thread.start()

# Tkinter GUI
root = tk.Tk()
root.title("Sign Language and Speech Recognition")
root.geometry("400x300")

deaf_button = tk.Button(root, text="Deaf Mode", command=lambda: run_in_thread(run_deaf), height=2, width=15)
deaf_button.pack(pady=20)

dumb_button = tk.Button(root, text="Dumb Mode", command=lambda: run_in_thread(run_dumb), height=2, width=15)
dumb_button.pack(pady=20)

exit_button = tk.Button(root, text="Exit", command=root.quit, height=2, width=15)
exit_button.pack(pady=20)

root.mainloop()