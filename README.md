# news2meme

This is the repository with the code for the paper "News2meme: An Automatic Content Generator from News Based on Word Subspaces from Text and Image".

Small notes:
- The work presented on the paper was obtained with a Matlab code which I cannot run anymore because I do not have Matlab's license anymore. Therefore, I re-wrote everything in Python and I hope this change makes the project more accessible (even though it has been over three years since I published this paper ^^")

- In the original code, I used pre-trained word2vec embeddings. However, in this repo, I opted for using pre-trained GLoVe just because it was a bit easier for me to work. I expect the results to be pretty similar for both.

## Installation

1. Clone this repo:
```
git clone https://github.com/ericashimomoto/news2meme.git
cd news2meme
```

2. Create a conda environment based on our dependecies and activate it:
```
conda env create -n <name> --file environment.yaml
conda activate <name>
```

Where you can replace `<name>` with whatever name you want.

3. Download everything:

    This script will download the following things in the folder `~/data/news2meme`:
    - The `glove.840B.300d.txt` pre-trained word embeddings.
    - The news2meme dataset, with the following structure and content:
        news2meme
            |-- catchphrases
                    |-- catchphrases.csv
            |-- memeimages
                    |-- full: Folder with the original images.
                    |-- memeImages.csv
            |-- news
                    |-- full: Folder with images used in each news article
                    |-- newsinlevels_level2.csv

