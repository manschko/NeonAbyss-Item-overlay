import requests
from bs4 import BeautifulSoup
import json
import os


# Function to download image
def download_image(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    response = requests.get(url)
    image_name = url.split('.png')[0].split('/')[-1] + '.png'
    with open(os.path.join(folder, image_name), 'wb') as file:
        file.write(response.content)
    return image_name[:-4]


# Function to scrape table
def scrape_table(url, folder):
    response = requests.get(url)
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


# URL of the website
url = 'https://neonabyss.fandom.com/wiki/Items'
scrape_table(url, 'img')
# Folder to