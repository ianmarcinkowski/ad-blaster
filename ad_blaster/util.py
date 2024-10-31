import io
from base64 import b64encode
from PIL import Image
import json


def ns_to_ms(nanoseconds: str | int):
    if isinstance(nanoseconds, str):
        nanoseconds = int(nanoseconds)
    return nanoseconds / 1000 / 1000 / 1000


def image_file_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        img_string = b64encode(buffered.getvalue()).decode("utf-8")
    return img_string

def find_json(message):
    start = message.find("{")
    end = message.find("}")
    found_json = None
    if start != -1 and end != -1:
        end += 1
        try:
            found_json = json.loads(message[start:end])
        except:
            found_json = None

    return found_json