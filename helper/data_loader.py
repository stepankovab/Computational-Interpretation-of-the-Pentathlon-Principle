import os
import json
import random

def data_loader(dataset_path, language = "en", without_repetition = False, shuffle = False, seed = 21):
    lyrics_path = os.path.join(dataset_path, 'HT.json')

    with open(lyrics_path, "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)

    if language == "en" or language == "cs":
        HT_lyrics = _extract_lyrics(dataset_dict, language)

    elif language == "both":
        HT_lyrics = _extract_lyrics(dataset_dict, "cs") + _extract_lyrics(dataset_dict, "en")

    else:
        HT_lyrics = []

    if without_repetition:
        HT_lyrics = list(dict.fromkeys(HT_lyrics))

    if shuffle:
        random.seed(seed)
        random.shuffle(HT_lyrics)

    return HT_lyrics

def _extract_lyrics(dataset_dict, language):
    list_of_sections = []

    for song in dataset_dict[language]:
        for section in dataset_dict[language][song]:
            list_of_sections.append(','.join(dataset_dict[language][song][section]))

    return list_of_sections

