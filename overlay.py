import json
import os
import tkinter as tk
from dataclasses import dataclass
from typing import Tuple, Dict
import concurrent.futures
import pytesseract
import re
from rapidfuzz import process, fuzz
import nltk
from nltk.corpus import words

import cv2
import numpy as np
from PIL import ImageGrab


# Load the list of English words
def load_custom_words():
    with open('db/dict', 'r') as f:
        return set(line.strip().lower() for line in f)


english_words = set(load_custom_words())
@dataclass
class ImageData:
    description: str
    # image_path: str
    # template: np.ndarray  # Store the template image
    # scale_factor: float = 1.0  # Scale factor compared to template
    # threshold: float = 0.55  # Match confidence threshold


class GameOverlay:
    def __init__(self, images_folder: str, data_file: str, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        self.label = None
        self.root = tk.Tk()
        self.images = {}
        self.text = ""
        self.load_image_data(images_folder, data_file)

    # Load the list of custom English words from a file
    def load_custom_words():
        with open('db/dict', 'r') as f:
            return set(line.strip().lower() for line in f)
        # self.setup_overlay_window()

    def load_image_data(self, images_folder: str, data_file: str):
        """Load images and their descriptions from the specified folder and data file"""


        # Load descriptions from JSON file
        with open(data_file, 'r') as f:
            descriptions = json.load(f)

        # Assign each key-value pair to self.images
        for key, value in descriptions.items():
            self.images[key] = ImageData(
                description=value
            )


        # # Load and process each image
        # for filename in os.listdir(images_folder):
        #     if filename.endswith(('.png', '.jpg', '.jpeg')):
        #         image_path = os.path.join(images_folder, filename)
        #         image_name = os.path.splitext(filename)[0]
        #
        #         if image_name in descriptions:
        #             # Load template image
        #             template = cv2.imread(image_path)
        #             self.images[image_name] = ImageData(
        #                 image_path=image_path,
        #                 description=descriptions[image_name],
        #                 template=template,
        #                 scale_factor=self.scale_factor
        #             )

    def setup_overlay_window(self):


        self.root.overrideredirect(True)  # Removes window decorations
        self.root.attributes("-topmost", True)  # Keeps the window on top of all others
        self.root.attributes("-transparentcolor", "black")  # Makes the background color transparent


        # Make the window full screen and transparent to clicks
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.config(bg='black')

        self.root.after(100, self._run)
        self.root.mainloop()

    def capture_screen(self) -> np.ndarray:
        """Capture the current screen content"""
        screen = np.array(ImageGrab.grab())
        return cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

    def detect_images(self, screen: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """Detect loaded images in the screen capture"""
        fixed_scale = 1.35  # Set your desired scale here
        detections = {}

        def process_image(name, img_data):
            template = cv2.imread(img_data.image_path)
            if template is None:
                return None

            resized_template = cv2.resize(template,
                                          (int(template.shape[1] * fixed_scale), int(template.shape[0] * fixed_scale)))
            result = cv2.matchTemplate(screen, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= img_data.threshold:
                return name, (max_loc[0], max_loc[1])
            return None

        max_threads = 6  # Set the maximum number of threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            results = list(executor.map(lambda item: process_image(*item), self.images.items()))

        for result in results:
            if result:
                name, loc = result
                detections[name] = loc

        return detections

    def detect_text(self, screen: np.ndarray):

        def customScorer(text, candidate, score_cutoff):
            if candidate.lower() in text.lower():
                return 100.0
            return fuzz.ratio(text, candidate)

        """Detect text in the screen capture"""
        detections = []
        height, width, _ = screen.shape
        top_crop = int(height * 0.2)  # Adjust the percentage as needed
        bottom_crop = int(height * 0.95)  # Adjust the percentage as needed
        left_crop = int(width * 0.1)  # Adjust the percentage as needed
        right_crop = int(width * 0.9)  # Adjust the percentage as needed
        cropped_screen = screen[top_crop:bottom_crop, left_crop:right_crop]
        # Convert screen to grayscale
        screen_gray = cv2.cvtColor(cropped_screen, cv2.COLOR_BGR2GRAY)


        # Apply binary threshold to keep only white and black colors
        _, screen_binary = cv2.threshold(screen_gray, 254, 255, cv2.THRESH_BINARY)

        lower_gray = np.array([120, 120, 120])
        upper_gray = np.array([135, 135, 135])
        gray_mask = cv2.inRange(cropped_screen, lower_gray, upper_gray)

        # Combine the binary threshold and gray mask
        combined_mask = cv2.bitwise_or(screen_binary, gray_mask)


        # # Display the processed image
        # cv2.imshow("Processed Image", combined_mask)
        # cv2.waitKey(0)  # Wait for a key press to close the window
        # cv2.destroyAllWindows()



        # Use pytesseract to detect text with custom configuration
        custom_config = r'--oem 3 --psm 6'
        text_data = pytesseract.image_to_data(combined_mask, config=custom_config, output_type=pytesseract.Output.DICT)

        for i, text in enumerate(text_data['text']):
            if text == '':
                continue

            # detections.append(text.strip())
            best_match, score, _ = process.extractOne(text.lower(), english_words, scorer=fuzz.ratio)
            if score < 80:
                continue
            detections.append(best_match.upper())
            # if text.strip() and text.strip().lower() in english_words:
            #     detections.append(text.strip())
        best_match, score, _ = process.extractOne(' '.join(detections), self.images.keys(), scorer=customScorer)
        # Combine all text fields with space and remove leading/trailing spaces
        if score < 80:
            return ''
        print(best_match + ' ' + str(score))
        return best_match

    def update_overlay(self, text):
        """Update overlay with detected images and their descriptions"""
        if self.text == text:
            return

        self.text = text
        if self.label:
            self.label.destroy()

        if text and text in self.images.keys():
            description = self.images[text].description
            self.label = tk.Label(self.root, text=description, font=('Arial', 15), fg='white', bg='black', wraplength=400)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.label.place(x=screen_width // 2, y=screen_height // 2, anchor='center')

    def run(self):
        self.setup_overlay_window()
        """Main loop for the overlay"""

    def _run(self):
        screen = self.capture_screen()
        detections = self.detect_text(screen)
        self.update_overlay(detections)
        self.root.after(1, self._run)



# Example usage:
if __name__ == "__main__":
    # Suggested data structure (save as data.json):
    """
    {
        "image1": "Description for image 1",
        "image2": "Description for image 2"
    }
    """
    overlay = GameOverlay(
        images_folder="./db/img/",
        data_file="./db/data.json"
    )
    overlay.run()
