import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import string
import torch
from torch.utils.data import Dataset, DataLoader


def get_tokens(sentence):

    raw_tokens = [word_tokenize(t) for t in sentence]

    stop_words = set(stopwords.words('english')) 

    tokens = []
    for t in raw_tokens:
        # lowercase
        lower_case = [w.lower() for w in t]
        
        # remove punctuation
        no_punct = [s.translate(str.maketrans('', '', string.punctuation)) for s in lower_case]
        
        filtered_sentence = [w for w in no_punct if not w in stop_words] 
        
        filtered_sentence[:] = [x for x in filtered_sentence if x]
        
        tokens.append(filtered_sentence)
    
    return tokens

class News2memeDataset(Dataset):

    def __init__(self, csv_file, type):
        self.anns = pd.read_csv(csv_file)
        self.type = type
    
    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        sample = self.anns.iloc[index]
        id = sample['id']

        if type == 'catchphrase':
            tokens = get_tokens(sample['catchPhrase']) + get_tokens(sample['character'])
        elif type == 'memeimage':
            tokens = get_tokens(sample['title'])
        elif type == 'news':
            tokens = get_tokens(sample['title']) + get_tokens(sample['text']) 

        return id, tokens

def load_wordembedding(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    word2vec = dict()
    for line in lines:
        line = line.strip('\n').split(' ')
        word = line[0]
        vec = [float(line[i]) for i in range(1,len(line))]
        vec = torch.tensor(vec, dtype = torch.float)
        word2vec[word] = vec

    return word2vec
    
def get_embeddings(tokens, word2vec, emb_dim=300):
    n_samples = len(tokens)
    lengths = [len(t) for t in tokens]
    max_length = max(lengths)

    embedding_matrix = torch.tensor(n_samples, emb_dim, max_length)

    for n, sample in enumerate(tokens):
        for t, token in enumerate(sample):
            embedding_matrix[n, :, t] = word2vec[token]
    
    return embedding_matrix, lengths

def make_dataloader(csv_file, type):
    dataset = News2memeDataset(csv_file, type)

    dataloader = DataLoader(dataset,
                                batch_size=16,
                                shuffle=False,
                                num_workers=4)
    
    return dataloader



