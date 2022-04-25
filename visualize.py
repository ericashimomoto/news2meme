import settings

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import pickle
import pandas as pd
import os

import argparse

def find_font_size(text, font, image, target_width_ratio):
    tested_font_size = 100
    tested_font = ImageFont.truetype(font, tested_font_size)
    observed_width, observed_height = get_text_size(text, image, tested_font)
    estimated_font_size = tested_font_size / (observed_width / image.width) * target_width_ratio
    return round(estimated_font_size)

def get_text_size(text, image, font):
    im = Image.new('RGB', (image.width, image.height))
    draw = ImageDraw.Draw(im)
    return draw.textsize(text, font)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--result', help='Pickle file containing the generation result by news2meme.')

    options = parser.parse_args()

    results = pickle.load(open(options.result, "rb"))

    images = pd.read_csv(settings.MEMEIMAGE_CSV)
    catchphrases = pd.read_csv(settings.CATCHPHRASE_CSV)

    if not os.path.exists(settings.VISUALIZATION_PATH):
        os.mkdir(settings.VISUALIZATION_PATH)

    for k in results.keys():
        img_id = results[k]['image_id']
        catchphrase_id = int(results[k]['catchphrase_id'])

        image_path = os.path.join(settings.MEMEIMAGE_PATH, images.iloc[k]['image_file'])
        catchphrase = catchphrases.iloc[catchphrase_id]['catchphrase']
        
        img = Image.open(image_path)
        I1 = ImageDraw.Draw(img)
        
        font_size = find_font_size(catchphrase, 'Impact.ttf', img, 0.8)

        font = ImageFont.truetype('Impact.ttf', font_size)
        I1.text((10, 10), catchphrase, font=font, fill =(255, 255, 255))

        #img.show()
        img_name = str(k) + ".jpeg"
        img.save(settings.VISUALIZATION_PATH + "/" + img_name)

        