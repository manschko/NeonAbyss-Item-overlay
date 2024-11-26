import concurrent.futures
import json
import tkinter as tk
from dataclasses import dataclass
from typing import Tuple, Dict

import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab
from rapidfuzz import process, fuzz


# Load words dict for item names
def load_custom_words():
    with open('db/dict', 'r') as f:
        return set(line.strip().lower() for line in f)


def custom_scorer(text, candidate, score_cutoff):
    if candidate.lower() in text.lower():
        return 100.0
    return fuzz.ratio(text, candidate)
item_words = set(load_custom_words())


@dataclass
class ItemData:
    description: str


def capture_screen() -> np.ndarray:
    """Capture the current screen content"""
    screen = np.array(ImageGrab.grab())
    return cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)


class GameOverlay:
    def __init__(self, data_file: str, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        self.label = None
        self.root = tk.Tk()
        self.data = {}
        self.text = ""
        self.load_data(data_file)

    def load_data(self, data_file: str):
        """Load images and their descriptions from the specified folder and data file"""

        # Load descriptions from JSON file
        with open(data_file, 'r') as f:
            self.data = json.load(f)

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
            results = list(executor.map(lambda item: process_image(*item), self.data.items()))

        for result in results:
            if result:
                name, loc = result
                detections[name] = loc

        return detections

    def detect_text(self, screen: np.ndarray):



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

        # Combine the binary threshold and gray mask to get white text from room and gray text from shop
        combined_mask = cv2.bitwise_or(screen_binary, gray_mask)

        # Use pytesseract to detect text with custom configuration
        custom_config = r'--oem 3 --psm 6'
        text_data = pytesseract.image_to_data(combined_mask, config=custom_config, output_type=pytesseract.Output.DICT)

        for i, text in enumerate(text_data['text']):
            if text == '':
                continue

            # detections.append(text.strip())
            best_match, score, _ = process.extractOne(text.lower(), item_words, scorer=fuzz.ratio)
            if score < 80:
                continue
            detections.append(best_match.upper())

        best_match, score, _ = process.extractOne(' '.join(detections), self.data.keys(), scorer=custom_scorer)
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

        if text and text in self.data.keys():
            description = self.data[text]
            self.label = tk.Label(self.root, text=description, font=('Arial', 15), fg='white', bg='black',
                                  wraplength=400)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.label.place(x=screen_width // 2, y=screen_height // 2, anchor='center')

    def run(self):
        self.setup_overlay_window()
        """Main loop for the overlay"""

    def _run(self):
        screen = capture_screen()
        detections = self.detect_text(screen)
        self.update_overlay(detections)
        self.root.after(1, self._run)


# Example usage:
if __name__ == "__main__":
    overlay = GameOverlay(
        data_file="./db/data.json"
    )
    overlay.run()
