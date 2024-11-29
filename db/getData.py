import requests
from bs4 import BeautifulSoup
import json
import os

DOWNLOAD_IMAGES = False #just in case in need the images
JSON_FILE = 'data.json'
DICT_NAME = 'dict'
# URL of the website
URL = 'https://neonabyss.fandom.com/wiki/Items'

def get_image_name(url):
    name = url.split('.png')[0].split('/')[-1] + '.png'
    name = name.replace('%27', "'")
    name = name.lower()
    name = name.replace('_', ' ')
    return name

# Function to download image
def download_image(url, folder):
    image_name = get_image_name(url)
    if DOWNLOAD_IMAGES:
        if not os.path.exists(folder):
            os.makedirs(folder)
        response = requests.get(url)
        with open(os.path.join(folder, image_name), 'wb') as file:
            file.write(response.content)


    return image_name[:-4]



def create_dict(data):

    # Create a set to store unique words
    unique_words = set()

    # Extract words from keys
    for key in data.keys():
        words = key.split(' ')
        for word in words:
            unique_words.add(word)

    with open(DICT_NAME, 'w') as f:
        for word in sorted(unique_words):
            f.write(word + '\n')

# Function to scrape table
def scrape_table(folder):
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table')  # Adjust selector based on table's location
    rows = tables[1].find_all('tr')

    data = {}

    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) < 2:
            continue  # Skip rows that don't have at least 2 columns

        image_url = cols[0].find('a')['href']  # Adjust selector based on image's location in column
        description = cols[1].text.strip()

        image_name = download_image(image_url, folder)
        data[image_name] = description

    with open('data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    create_dict(data)



scrape_table('img')
# Folder to