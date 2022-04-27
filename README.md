# news2meme

This is the repository with the code for the paper "News2meme: An Automatic Content Generator from News Based on Word Subspaces from Text and Image".

Small notes:
- The work presented on the paper was obtained with a Matlab code which I cannot run because I do not have Matlab's license anymore. Therefore, I re-wrote everything in Python and I hope this change makes the project more accessible (even though it has been over three years since I published this paper ^^")

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

- Only got to test this code on Mac. It should run with not much trouble on Linux. I have no idea if it will run properly on Windows (sorry!).

3. Download everything:

    This script will download the following things in the folder `~/data/news2meme`:
    - The `glove.42B.300d.txt` pre-trained word embeddings.
    - The news2meme dataset, with the following structure and content:
    
            |-- news2meme
                |-- catchphrases
                    |-- catchphrases.csv
                |-- memeimages
                    |-- full: Folder with the original images.
                    |-- memeImages.csv
                |-- news
                    |-- full: Folder with the images available with each news article.
                    |-- newsinlevels_level2.csv
    
    For News2meme, you don't actually need the images from the articles, but I included in the package as it might be useful for someone playing with the small dataset :)

### Configuration

If you changed the download folder, make sure to change the path in settings.py

## Train Mode

To be able to generate the memes, you first need to compute the subspaces for each meme image and each catchphrase in the dataset. To do this, run the following:

    `python news2meme.py --config=experiments/train.yaml`

This code will save all the subspaces into the `OUTPUT_PATH` defined in `settings.py`. I set the contribution ratio to define the subspace dimensions to `0.5`, but feel free to change this value in the yaml file to see how it changes the results.

## Generation Mode

You can generate memes in three different ways:

1. Free mode: Put your news article in a .txt file and modify the `experiments/gen_single.yaml` file with the path of the file in the `TEXT` field. Then, run the following:

    `python news2meme.py --config=experiments/gen_single.yaml`

2. Specific mode: Define the news article ID in the `experiments/gen_id.yaml` file in the `NEWS_ID` field. Then, run the following:

    `python news2meme.py --config=experiments/gen_id.yaml`
    
3. Random mode: Generate a meme for a randomly sampled news article. Run the following:
    `python news2meme.py --config=experiments/gen_random.yaml`
    
4. Full mode: Generate a meme for each news article in the database. Run the following:
    `python news2meme.py --config=experiments/gen_all.yaml`

The news2meme will place a pickle file with the retrieved meme image and catchphrase id in the path specified by `OUTPUT_PATH` in `settings.py`. Again, I set the contribution ratio to be `0.8`, but feel free to change this value in the respective yaml files to see how it changes the results.

## Visualization

To visualize the generated memes, run the following:

`python visualize.py --result=output/gen_memes_ids.pickle`

Where `output/gen_memes_ids.pickle` specifies the results generated by news2meme. The generated memes will be saved in the folder specified in field `VISUALIZATION_PATH` in `settings.py`.

## Citation

If you use our code or dataset, please consider citing our work:

```bibtex
@inproceedings{shimomoto2019news2meme,
  title={News2meme: An automatic content generator from news based on word subspaces from text and image},
  author={Shimomoto, Erica K and Souza, Lincon S and Gatto, Bernardo B and Fukui, Kazuhiro},
  booktitle={2019 16th International Conference on Machine Vision Applications (MVA)},
  pages={1--6},
  year={2019},
  organization={IEEE}
} 
```

## Questions?

If you have any questions or problems while running this code, feel free to open an issue or to contact me at kidoshimomoto.e@aist.go.jp
