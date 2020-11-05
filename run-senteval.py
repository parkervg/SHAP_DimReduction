"""
Adapted from SentEval/examples/bow.py
https://github.com/facebookresearch/SentEval/blob/master/examples/bow.py
"""
import sys
import io
import numpy as np
import argparse
import logging
from tools.Blogger import Blogger
logger = Blogger()

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec

# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = params.word_vec["the"].shape[0]
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SentEval runner"
    )
    parser.add_argument("-tests", nargs="+", help="The tests to run")
    parser.add_argument("-path", type=str, help="The path to the vectors to load")
    parser.add_argument("-pytorch", type=str2bool, default=False, help="Whether to use pytorch as classifier")
    parser.add_argument("-batch_size", type=int, default=128, help="Batch size for classification")
    parser.add_argument("-epoch_size", type=int, default=2, help="Epoch size")
    values = parser.parse_args()
    logger.green(values)
    PATH_TO_VEC = values.path
    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': values.pytorch, 'kfold': 5}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': values.batch_size,
                                     'tenacity': 3, 'epoch_size': values.epoch_size}
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    result = se.eval(values.tests)
    for k in result:
        logger.status_update("{}: {}".format(k, result[k]["acc"]))
        print()
