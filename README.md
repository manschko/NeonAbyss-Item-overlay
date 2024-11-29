<!-- Title -->
<div align="center">
  <h1>Neon Abyss Item Overlay</h1>
  
  <p>
    An overlay for the game Neon Abyss that reads the screen for item names and displays their descriptions.
  </p>

</div>

<br />

<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
  * [Usage](#usage)
- [Getting Started](#getting-started)
  * [Tech Stack](#tech-stack)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Run Locally](#run-locally)
  

<!-- About the Project -->
## About the Project

[//]: # (<div align="center">)

[//]: # (  <img src="https://placehold.co/600x400?text=Your+GIF+here" alt="how it works" />)

[//]: # (</div>)

Neon Abyss Item Overlay is an overlay for the game Neon Abyss that reads the screen for item names and displays their descriptions on the screen.

### Usage

1. Download the latest release for your platform from the [releases page](https://github.com/mtruckses/neon-abyss-item-overlay/releases).
2. Uncompress the zip file.
3. Ensure Tesseract OCR is installed and in the PATH. 
4. Start the overlay.



<!-- Getting Started -->
## 	Getting Started

<!-- TechStack -->
### Tech Stack

<details>
  <summary>Languages</summary>
  <ul>
    <li><a href="https://www.python.org/">Python</a></li>
  </ul>
</details>

<details>
  <summary>Libraries</summary>
  <ul>
    <li><a href="https://opencv.org/">OpenCV</a></li>
    <li><a href="https://pillow.readthedocs.io/">Pillow</a></li>
    <li><a href="https://pypi.org/project/pytesseract/">pytesseract</a></li>
    <li><a href="https://github.com/maxbachmann/RapidFuzz">RapidFuzz</a></li>
    <li><a href="https://numpy.org/">NumPy</a></li>
  </ul>
</details>

### Prerequisites

This project requires the following software to be installed:

- Python 3.x
- Tesseract OCR

<!-- Installation -->
### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/mtruckses/neon-abyss-item-overlay.git
    cd neon-abyss-item-overlay
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure Tesseract OCR is installed and added to your system's PATH. You can download it from [here](https://github.com/tesseract-ocr/tesseract).

### Run Locally

1. Start the overlay:
    ```sh
    python overlay.py
    ```
<!-- Running Tests -->

[//]: # (### Running Tests)

[//]: # ()
[//]: # (To run tests, use the following command:)

[//]: # (    ```sh)

[//]: # (    # Add your test command here)

