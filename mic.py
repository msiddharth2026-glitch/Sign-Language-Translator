import speech_recognition as sr
import cv2
from string import ascii_lowercase

# Alphabet mapping to positions
LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)}

# Function to get the position of letters
def alphabet_position(text):
    text = text.lower()
    numbers = [LETTERS[character] for character in text if character in LETTERS]
    return numbers  # Returns a list of positions as strings

# Function to resize images to a uniform size
def resize_image(image, width=50, height=50):
    return cv2.resize(image, (width, height))

# Speech recognizer setup
r = sr.Recognizer()
speech = sr.Microphone(device_index=1)

# Listen for speech input
with speech as source:
    print("say something!…")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

import numpy as np  # Add this import at the beginning of your code

try:
    # Recognize speech
    recog = r.recognize_google(audio, language='en-US')
    print("You said: " + recog)

    # Process each character in the recognized text, including spaces
    images = []
    for char in recog:
        if char == " ":  # If a space is found, add the space image
            img_path = 'audio/space.jpg'
        elif char.lower() in LETTERS:
            l = alphabet_position(char)[0]  # Get the letter's position
            k = int(l) - 1  # Adjust to zero-based index
            img_path = 'audio/' + str(k) + '.jpg'
        else:
            continue  # Skip non-alphabetic characters

        # Read the image
        image = cv2.imread(img_path)

        if image is not None:
            resized_image = resize_image(image)  # Resize the image to uniform size
            images.append(resized_image)  # Store the resized images in a list
        else:
            print(f"Image for {char} not found at {img_path}")

    if images:
        # Parameters for grid layout
        images_per_row = 10  # Number of images per row
        row_images = []  # List to hold rows of images
        current_row = []  # Temporary storage for the current row

        for img in images:
            current_row.append(img)
            if len(current_row) == images_per_row:  # If the row is full
                row_images.append(cv2.hconcat(current_row))  # Horizontally stack the row
                current_row = []  # Start a new row

        if current_row:  # Add the remaining images in the last row, if any
            # Fill the row with blank images to maintain consistent width
            while len(current_row) < images_per_row:
                blank_image = resize_image(255 * np.ones_like(images[0]))  # Create a blank image
                current_row.append(blank_image)
            row_images.append(cv2.hconcat(current_row))

        # Vertically stack all rows to form the final image
        combined_image = cv2.vconcat(row_images)

        # Display the combined image
        window_name = 'Detected Characters'
        cv2.imshow(window_name, combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid images found.")


except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
