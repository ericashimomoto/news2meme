# !/usr/bin bash
#
DOWNLOAD_PATH=~/data/News2Meme
mkdir -p $DOWNLOAD_PATH
mkdir $DOWNLOAD_PATH/word_embeddings

cd $DOWNLOAD_PATH
# download pre-trained GLoVe word embeddings
wget -P $DOWNLOAD_PATH/word_embeddings http://nlp.stanford.edu/data/glove.42B.300d.zip 
unzip $DOWNLOAD_PATH/word_embeddings/glove.42B.300d.zip

# download dataset
wget -P $DOWNLOAD_PATH https://www.dropbox.com/s/4au9bkv5xzi79ix/News2Meme.zip
unzip $DOWNLOAD_PATH/News2Meme.zip