import subspace
import data
import settings

import torch
import torch.nn.functional as F
import yaml
import pickle
import os
import random

import argparse

def compute_subspaces(word2vec, contRatio):

    memeimage_dataloader = data.make_dataloader(settings.MEMEIMAGE_CSV, "memeimage")
    catchphrase_dataloader = data.make_dataloader(settings.CATCHPHRASE_CSV, "catchphrase")
    
    memeimage_subspaces = []
    memeimage_subdims = []
    memeimage_ids = []
    catchphrase_subspaces = []
    catchphrase_subdims = []
    catchphrase_ids = []

    for i, batch in enumerate(memeimage_dataloader):
        ids = batch[0]
        tokens = batch[1]
        lens = batch[2]
        
        embeddings = data.get_embeddings(tokens, word2vec)
        
        subspaces, dims = subspace.subspace_bases(embeddings, lens, contRatio)

        memeimage_subspaces.append(subspaces)
        memeimage_subdims.append(dims)
        memeimage_ids.append(ids)
    
    for batch in catchphrase_dataloader:
        ids = batch[0]
        tokens = batch[1]
        lens = batch[2]

        embeddings = data.get_embeddings(tokens, word2vec)
        
        subspaces, dims = subspace.subspace_bases(embeddings, lens, contRatio)

        catchphrase_subspaces.append(subspaces)
        catchphrase_subdims.append(dims)
        catchphrase_ids.append(ids)

    memeimage_subdims = torch.cat(memeimage_subdims)
    memeimage_ids = torch.cat(memeimage_ids)
    catchphrase_subdims = torch.cat(catchphrase_subdims)
    catchphrase_ids = torch.cat(catchphrase_ids)

    max_memeimage_subdim= torch.max(memeimage_subdims)
    max_catchphrase_subdim = torch.max(catchphrase_subdims)

    for i in range(len(memeimage_subspaces)):
        diff = max_memeimage_subdim - memeimage_subspaces[i].shape[2]
        if diff > 0:
            memeimage_subspaces[i] = F.pad(input=memeimage_subspaces[i], pad = (0,diff,0,0), mode='constant', value=0)
    
    memeimage_subspaces = torch.cat(memeimage_subspaces, dim=0)

    for i in range(len(catchphrase_subspaces)):
        diff = max_catchphrase_subdim - catchphrase_subspaces[i].shape[2]
        if diff > 0:
            catchphrase_subspaces[i] = F.pad(input=catchphrase_subspaces[i], pad = (0,diff,0,0), mode='constant', value=0)
    
    catchphrase_subspaces = torch.cat(catchphrase_subspaces, dim=0)

    return memeimage_ids, memeimage_subspaces, memeimage_subdims, catchphrase_ids, catchphrase_subspaces, catchphrase_subdims


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Yaml file with configuration.')

    options = parser.parse_args()

    with open(options.config, "r") as stream:
        config = yaml.safe_load(stream)

    print('Loading WE.')
    word2vec = data.load_wordembedding(settings.EMBEDDINGS_PATH)

    if config['MODE'] == 'TRAIN':
        print('Train mode.')

        print('Computing subspaces...')
        memeimage_ids, memeimage_subspaces, memeimage_subdims, catchphrase_ids, catchphrase_subspaces, catchphrase_subdims = compute_subspaces(word2vec, config['CRATIO'])

        meme_subspaces = {'ids': memeimage_ids, 'subspaces': memeimage_subspaces, 'subdims':memeimage_subdims}
        catch_subspaces = {'ids': catchphrase_ids, 'subspaces':catchphrase_subspaces, 'subdims':catchphrase_subdims}

        if not os.path.exists(settings.OUTPUT_PATH):
            os.mkdir(settings.OUTPUT_PATH)
        print('Saving...')
        with open(settings.OUTPUT_PATH + '/meme_subspaces.pickle', "wb") as f:
            pickle.dump(meme_subspaces, f)

        with open(settings.OUTPUT_PATH + '/catch_subspaces.pickle', "wb") as f:
            pickle.dump(catch_subspaces, f)
        print('Done!')

    if config['MODE'] == 'GEN':
        print('Generation mode.')
        ids = []
        if 'SUBSPACE_PATH' in config.keys():
            subspace_path = config['SUBSPACE_PATH']
        else:
            subspace_path = settings.OUTPUT_PATH
        
        # Load subspaces
        with open(settings.OUTPUT_PATH + '/meme_subspaces.pickle', "rb") as f:
            meme_subspaces = pickle.load(f)

        with open(settings.OUTPUT_PATH + '/catch_subspaces.pickle', "rb") as f:
            catch_subspaces = pickle.load(f)
        
        news = []
        if config['GEN_MODE'] == 'FREE':
            print('Input: ' + config['TEXT'])
            with open(config['TEXT'], "r") as f:
                text = f.readlines()
            text = ' '.join(text)
            new = data.get_tokens(text, True)
            news.append(new)
            ids.append(0)
        else:
            news_dataset = data.News2memeDataset(settings.NEWS_CSV, 'news')
            
            if config['GEN_MODE'] == 'SPECIFIC':
                print('Input: news ' + str(config['NEWS_ID']))
                ids.append(config['NEWS_ID'])
                _, new = news_dataset[config['NEWS_ID']]
                news.append(new)

            elif config['GEN_MODE'] == 'RANDOM':
                id = random.randint(0, len(news_dataset))
                print('Input: random, news ' + str(id))
                ids.append(id)
                _, new = news_dataset[id]
                news.append(new)

            elif config['GEN_MODE'] == 'ALL':
                print('Input: All news!')
                n_news = len(news_dataset)
                for i in range(n_news):
                    id, new = news_dataset[i]
                    ids.append(id)
                    news.append(new)
        
        padded_news, news_lens = data.pad_to_maxlen(news)
        news_embeddings = data.get_embeddings(padded_news, word2vec)

        n_subspaces, n_dims = subspace.subspace_bases(news_embeddings, news_lens, config['CRATIO'])

        print('Searching for best meme image and best cathphrase...')
        closest_images = subspace.find_closest_subspace(meme_subspaces['subspaces'], meme_subspaces['subdims'], n_subspaces, n_dims)
        closest_catchphrases = subspace.find_closest_subspace(catch_subspaces['subspaces'], catch_subspaces['subdims'], n_subspaces, n_dims)
        
        result = {}
        for i in range(len(ids)):
            r = {'image_id':meme_subspaces['ids'][closest_images[i]], 'catchphrase_id':catch_subspaces['ids'][closest_catchphrases[i]]}
            result[ids[i]] = r

        print('Saving pickle results!')
        if not os.path.exists(settings.OUTPUT_PATH):
            os.mkdir(settings.OUTPUT_PATH)

        with open(settings.OUTPUT_PATH + '/gen_memes_ids.pickle', "wb") as f:
            pickle.dump(result, f)
