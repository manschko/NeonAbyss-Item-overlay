import unittest
import cv2
import os
from overlay import GameOverlay


class text_recognition_test(unittest.TestCase):
    def setUp(self):
        os.chdir('..')  # Change the working directory to the parent folder
        self.overlay = GameOverlay(data_file='./db/data.json')

    def test_detect_text(self):
        images_folder = './test/images'
        for image_name in os.listdir(images_folder):
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_folder, image_name)
                screen = cv2.imread(image_path)
                detected_text = self.overlay.detect_text(screen)
                expected_text = os.path.splitext(image_name)[0].upper()
                self.assertEqual(detected_text, expected_text, f"Failed for image: {image_name}")


if __name__ == '__main__':
    unittest.main()