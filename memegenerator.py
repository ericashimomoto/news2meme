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
    catchphrase_subspaces = []
    catchphrase_subdims = []

    import pdb; pdb.set_trace()
    for i, batch in enumerate(memeimage_dataloader):
        ids = batch[0]
        tokens = batch[1]
        
        embeddings, lenghts = data.get_embeddings(tokens, word2vec)
        
        subspaces, dims = subspace.subspace_bases(embeddings, torch.tensor(lenghts), contRatio)

        memeimage_subspaces.append(subspaces)
        memeimage_subdims.append(dims)
    
    for batch in catchphrase_dataloader:
        ids = batch[0]
        tokens = batch[1]
        
        embeddings, lenghts = data.get_embeddings(tokens, word2vec)
        
        subspaces, dims = subspace.subspace_bases(embeddings, torch.tensor(lenghts), contRatio)

        catchphrase_subspaces.append(subspaces)
        catchphrase_subdims.append(dims)

    memeimage_subdims = torch.cat(memeimage_subdims)
    catchphrase_subdims = torch.cat(catchphrase_subdims)

    max_memeimage_subdim= torch.max(memeimage_subdims)
    max_catchphrase_subdim = torch.max(catchphrase_subdims)

    for i in range(len(memeimage_subspaces)):
        diff = max_memeimage_subdim - memeimage_subdims[i]
        if diff > 0:
            memeimage_subspaces[i] = F.pad(input=memeimage_subspaces[i], pad = (0,diff,0,0), mode='constant', value=0)
    
    memeimage_subspaces = torch.stack(memeimage_subspaces)

    for i in range(len(catchphrase_subspaces)):
        diff = max_catchphrase_subdim - catchphrase_subdims[i]
        if diff > 0:
            catchphrase_subspaces[i] = F.pad(input=catchphrase_subspaces[i], pad = (0,diff,0,0), mode='constant', value=0)
    
    catchphrase_subspaces = torch.stack(catchphrase_subspaces)

    return memeimage_subspaces, memeimage_subdims, catchphrase_subspaces, catchphrase_subdims


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
        memeimage_subspaces, memeimage_subdims, catchphrase_subspaces, catchphrase_subdims = compute_subspaces(word2vec, config['CRATIO'])

        meme_subspaces = {'subspaces': memeimage_subspaces, 'subdims':memeimage_subdims}
        catch_subspaces = {'subspaces':catchphrase_subspaces, 'subdims':catchphrase_subdims}

        if not os.path.exists(settings.OUTPUT_PATH):
            os.mkdir(settings.OUTPUT_PATH)
        print('Saving...')
        with open(settings.OUTPUT_PATH + '/meme_subspaces.pickle', "wb") as f:
            pickle.dump(meme_subspaces, f)

        with open(settings.OUTPUT_PATH + '/catch_subspaces.pickle', "wb") as f:
            pickle.dump(catch_subspaces, f)
        print('Done!')

    if config['MODE'] == 'GEN':
        ids = []
        if config.has_key('SUBSPACE_PATH'):
            subspace_path = config['SUBSPACE_PATH']
        else:
            subspace_path = settings.OUTPUT_PATH
        
        # Load subspaces
        with open(settings.OUTPUT_PATH + '/meme_subspaces.pickle', "rb") as f:
            meme_subspaces = pickle.load(f)

        with open(settings.OUTPUT_PATH + '/catch_subspaces.pickle', "wb") as f:
            catch_subspaces = pickle.load(f)
        
        news = []
        if config['GEN_MODE'] == 'FREE':
            new = data.get_tokens(config['TEXT'])
            news.append(new)
        else:
            news_dataset = data.News2memeDataset(settings.NEWS_CSV, 'news')
            
            if config['GEN_MODE'] == 'SPECIFIC':
                ids.append(config['NEWS_ID'])
                _, new = news_dataset[config.NEWS_ID]
                news.append(new)
            elif config['GEN_MODE'] == 'RANDOM':
                id = random.randint(0, len(news_dataset))
                ids.append(id)
                _, new = news_dataset[news_dataset[id]]
                news.append(new)
            elif config['GEN_MODE'] == 'ALL':
                n_news = len(news_dataset)
                for i in range(n_news):
                    id, new = news_dataset[i]
                    ids.append(id)
                    news.append(new)
        
        news_embeddings = data.get_embeddings(news, word2vec)

        n_subspaces, n_dims = subspace.subspace_bases(news_embeddings, config['CRATIO'])

        closest_images = subspace.find_closest_subspace(meme_subspaces['subspaces'], meme_subspaces['subdims'], n_subspaces, n_dims)
        closest_catchphrases = subspace.find_closest_subspace(catch_subspaces['subspaces'], meme_subspaces['subdims'], n_subspaces, n_dims)

        result = {}
        for i in range(len(ids)):
            r = {'image':closest_images[i], 'catchphrase':closest_catchphrases[i]}
            result[ids[i]] = r

        if not os.path.exists(settings.OUTPUT_PATH):
            os.mkdir(settings.OUTPUT_PATH)

        with open(settings.OUTPUT_PATH + '/gen_memes_ids.pickle', "wb") as f:
            pickle.dump(meme_subspaces, f)
