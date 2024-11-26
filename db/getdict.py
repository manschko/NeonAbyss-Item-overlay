import json
import os

def extract_words_from_keys(json_file, output_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create a set to store unique words
    unique_words = set()

    # Extract words from keys
    for key in data.keys():
        words = key.split('_')
        for word in words:
            unique_words.add(word.lower())

    with open(output_file, 'w') as f:
        for word in sorted(unique_words):
            f.write(word + '\n')

# Example usage
json_file = 'data.json'
output_folder = 'dict'
extract_words_from_keys(json_file, output_folder)