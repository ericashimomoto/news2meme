import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import string
import torch
from torch.utils.data import Dataset, DataLoader


def get_tokens(sentence, filter_stopwords=False):
    raw_tokens = word_tokenize(sentence)

    # lowercase
    lower_case = [w.lower() for w in raw_tokens]
    
    # remove punctuation
    filtered_sentence = [s.translate(str.maketrans('', '', string.punctuation)) for s in lower_case]
    
    if filter_stopwords:
        stop_words = set(stopwords.words('english')) 
        filtered_sentence = [w for w in filtered_sentence if not w in stop_words] 
    
    # Remove empty tokens
    filtered_sentence[:] = [x for x in filtered_sentence if x]

    return filtered_sentence

class News2memeDataset(Dataset):

    def __init__(self, csv_file, type):
        if type == 'news':
            self.anns = pd.read_csv(csv_file, sep=";")
        else:
            self.anns = pd.read_csv(csv_file)
        self.type = type
    
    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        sample = self.anns.iloc[index]
        id = sample['id']

        if self.type == 'catchphrase':
            tokens = get_tokens(sample['catchphrase']) 
            if type(sample['character']) == str:
                tokens += get_tokens(sample['character'])
            if type(sample['mediasource']) == str:
                tokens += get_tokens(sample['mediasource'])
        elif self.type == 'memeimage':
            tokens = get_tokens(sample['title'])
        elif self.type == 'news':
            tokens = get_tokens(sample['title']) + get_tokens(sample['text'], True) 

        return id, tokens

def pad_to_maxlen(tokens):

    lens = torch.zeros(len(tokens), dtype=torch.uint8)
    batch_max_len = 0
    for i, t in enumerate(tokens):
        lens[i] = len(t)
        batch_max_len = max(batch_max_len, len(t))
        
    # Pad tokens to max length
    padded_tokens = []
    for i, t in enumerate(tokens):
        padded_t = t + ['<PAD>'] * (batch_max_len-lens[i])
        padded_tokens.append(padded_t) 
    
    return padded_tokens, lens
        

def collate_fn(data):
    ids, tokens = zip(*data)

    ids = torch.tensor(ids)

    padded_tokens, lens = pad_to_maxlen(tokens)

    return ids, padded_tokens, lens
        
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

    word2vec['<PAD>'] = torch.zeros(vec.shape)
    return word2vec
    
def get_embeddings(tokens, word2vec, emb_dim=300):
    n_samples = len(tokens)
    max_length = len(tokens[0])

    embedding_matrix = torch.zeros(n_samples, emb_dim, max_length)

    for n, sample in enumerate(tokens):
        for t, token in enumerate(sample):
            if token in word2vec.keys():
                embedding_matrix[n, :, t] = word2vec[token]
            else:
                embedding_matrix[n, :, t] = torch.zeros(emb_dim)
    
    return embedding_matrix

def make_dataloader(csv_file, type):
    dataset = News2memeDataset(csv_file, type)

    dataloader = DataLoader(dataset,
                                batch_size=16,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=collate_fn)
    
    return dataloader



