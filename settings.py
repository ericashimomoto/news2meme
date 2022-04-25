import os

CODE_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_PATH='/Users/ericaks/Datasets/News2Meme'

CATCHPHRASE_CSV = os.path.join(DATA_PATH, "catchphrases/catchphrases.csv")
MEMEIMAGE_CSV = os.path.join(DATA_PATH, "memeimages/memeImages.csv")
NEWS_CSV = os.path.join(DATA_PATH, "news/newsinlevels_level2.csv")

EMBEDDINGS_PATH = '/Users/ericaks/Datasets/GloVe/glove.42B.300d.txt'

OUTPUT_PATH = os.path.join(CODE_ROOT, "output")
