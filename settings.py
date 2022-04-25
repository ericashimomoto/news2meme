import os

CODE_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = '~/data/news2meme' #'/Users/ericaks/Datasets/News2Meme'

CATCHPHRASE_CSV = os.path.join(DATA_PATH, "News2Meme/catchphrases/catchphrases.csv")
MEMEIMAGE_CSV = os.path.join(DATA_PATH, "News2Meme/memeimages/memeImages.csv")
MEMEIMAGE_PATH = os.path.join(DATA_PATH, "News2Meme/memeimages")
NEWS_CSV = os.path.join(DATA_PATH, "News2Meme/news/newsinlevels_level2.csv")

EMBEDDINGS_PATH = os.path.join(DATA_PATH, 'word_embeddings/glove.42B.300d.txt') #'/Users/ericaks/Datasets/GloVe/glove.42B.300d.txt'

OUTPUT_PATH = os.path.join(CODE_ROOT, "output")

VISUALIZATION_PATH = os.path.join(OUTPUT_PATH, "visualization")
