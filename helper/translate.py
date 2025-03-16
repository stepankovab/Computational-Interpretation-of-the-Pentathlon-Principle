import requests
import time
import re

def lindat_translate(text_list, from_lang, to_lang, separator):
    succes = False
    start_time = time.time()
    url = f'http://lindat.mff.cuni.cz/services/translation/api/v2/models/{from_lang}-{to_lang}'
    output = ""

    while not succes and (time.time() - start_time) < 10:
        response = requests.post(url, data = {"input_text": separator.join(text_list)})
        response.encoding='utf8'
        output = response.text[:-1]

        if re.match("request", output):
            output = separator.join(text_list)
            print("saved ya")
            time.sleep(1)
        else:
            succes = True

    return output