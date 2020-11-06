from sklearn.decomposition import PCA
import numpy as np
from tools.Blogger import Blogger
import tensorflow.compat.v1 as tf
from tensorflow_fcwta.models import FullyConnectedWTA
import json
import io
import imp
import os
import uuid
import sys
EPSILON = 1e-6
RAND_STATE = 324
logger = Blogger()

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

"""
TODO:
    - Provide 'inference' argument on se.eval class to not return stats, only model and data
"""


class WordEmbeddings:

    def __init__(self,
                 vector_file,
                 normalize_on_load=False):
        self.vector_file = vector_file if vector_file else "./embeds/glove.6B.300d.txt"
        # On normalization:
        # Levy et. al. 2015
        #   "Vectors are normalized to unit length before they are used for similarity calculation,
        #    making cosine similarity and dot-product equivalent.""
        self.normalize_on_load = normalize_on_load
        self.prev_components = np.empty((0,0))
        self.function_log = []
        self.load_word_vectors()

    def load_word_vectors(self):
        logger.status_update("Loading vectors at {}...".format(self.vector_file))
        self.ordered_vocab = []
        self.embeds = []
        with io.open(self.vector_file, 'r', encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ',1)
                self.ordered_vocab.append(word)
                self.embeds.append(np.fromstring(vec, sep=' '))
                if self.normalize_on_load:
                    self.embeds[-1] /= math.sqrt((self.embeds[-1]**2).sum() + EPSILON)
        self.original_dim = len(self.embeds[0])
        self.embeds = np.asarray(self.embeds)

    def pca_fit_transform(self, output_dims):
        self.function_log.append("pca_fit_transform")
        pca = PCA(n_components=output_dims, random_state=RAND_STATE)
        self.embeds = pca.fit_transform(self.embeds)
        self.prev_components = pca.components_

    def pca_fit(self):
        self.function_log.append("pca_fit")
        pca = PCA(n_components=len(self.embeds[0]))
        pca.fit(self.embeds)
        self.prev_components = pca.components_

    def remove_top_components(self, k):
        self.function_log.append("remove_top_components")
        if self.prev_components.size == 0:
            raise ValueError("No value found for prev_components. Did you call pca_fit_transform?")
        z = []
        for ix, x in enumerate(self.embeds):
            for u in self.prev_components[0:k]:
                x = x - np.dot(u.transpose(), x) * u
            z.append(x)
        self.embeds = np.asarray(z)

    def subract_mean(self):
        self.function_log.append("subract_mean")
        self.embeds = self.embeds - np.mean(self.embeds)

    def sparsify(self, checkpoint_path):
        """
        Calls a Tensoflow checkpoint of a fcwta training instance to sparsify embeddings
        """
        self.function_log.append("sparsify")
        flags_file = imp.load_source("flags.py", os.path.dirname(checkpoint_path) + "/flags.py")
        with tf.Session() as sess:
            fcwta = FullyConnectedWTA(self.original_dim,
                          flags_file.FLAGS.batch_size,
                          sparsity=flags_file.FLAGS.sparsity,
                          hidden_units=flags_file.FLAGS.hidden_units,
                          encode_layers=flags_file.FLAGS.num_layers,
                          learning_rate=flags_file.FLAGS.learning_rate)
            fcwta.saver.restore(sess, checkpoint_path)
            self.embeds = fcwta.encode(sess, self.embeds)


    # Create dictionary
    def _create_dictionary(self, sentences, threshold=0):
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


    # SentEval prepare and batcher
    def prepare(self, params, samples):
        _, params.word2id = self._create_dictionary(samples)
        params.word_vec = self._load_eval_vectors(params.word2id)
        params.wvec_dim = params.word_vec["the"].shape[0]
        return

    def _load_eval_vectors(self, word2id):
        word_vec = {}
        for word, embed in zip(self.ordered_vocab, self.embeds):
            if word in word2id:
                word_vec[word] = embed
        logger.log('Found {0} words with word vectors, out of \
            {1} words'.format(len(word_vec), len(word2id)))
        return word_vec

    @staticmethod
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


    def SentEval(self, tasks, save_summary=False, summary_file_name=None):
        # Set params for SentEval
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare)
        results = se.eval(tasks)
        for k in results:
            summary["scores"][k] = results[k]["acc"]
            logger.status_update("{}: {}".format(k, results[k]["acc"]))
            print()
        if save_summary:
            summary = {}
            summary["original_dim"] = self.original_dim
            summary["final_dim"] = self.embeds.shape[1]
            summary["process"] = self.function_log
            summary["scores"] = {}
            summary_file_name = summary_file_name if summary_file_name else str(uuid.uuid4())
            self.save_summary_json(summary, summary_file_name)

    @staticmethod
    def save_summary_json(summary, summary_file_name):
        if not os.path.isdir("summary"):
            os.mkdir("summary")
        with open("summary/{}".format(summary_file_name), 'w') as f:
            json.dump(summary, f)
        logger.status_update("Summary saved to summary/{}".format(summary_file_name))

    def save_vectors(self, output_file):
        vector_size = self.embeds.shape[1]
        assert (len(self.ordered_vocab), vector_size) == self.embeds.shape
        with open(output_file, "w", encoding="utf-8") as out:
            for ix, word in enumerate(self.ordered_vocab):
                out.write("%s " % word)
                for t in self.embeds[ix]:
                    out.write("%f " % t)
                out.write("\n")
        logger.status_update("Vectors saved to {}".format(output_file))


# # Create dictionary
# def create_dictionary(sentences, threshold=0):
#     words = {}
#     for s in sentences:
#         for word in s:
#             words[word] = words.get(word, 0) + 1
#
#     if threshold > 0:
#         newwords = {}
#         for word in words:
#             if words[word] >= threshold:
#                 newwords[word] = words[word]
#         words = newwords
#     words['<s>'] = 1e9 + 4
#     words['</s>'] = 1e9 + 3
#     words['<p>'] = 1e9 + 2
#
#     sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
#     id2word = []
#     word2id = {}
#     for i, (w, _) in enumerate(sorted_words):
#         id2word.append(w)
#         word2id[w] = i
#
#     return id2word, word2id
#
# # SentEval prepare and batcher
# def prepare(params, samples):
#     _, params.word2id = create_dictionary(samples)
#     params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
#     params.wvec_dim = params.word_vec["the"].shape[0]
#     return
#
# def batcher(params, batch):
#     batch = [sent if sent != [] else ['.'] for sent in batch]
#     embeddings = []
#
#     for sent in batch:
#         sentvec = []
#         for word in sent:
#             if word in params.word_vec:
#                 sentvec.append(params.word_vec[word])
#         if not sentvec:
#             vec = np.zeros(params.wvec_dim)
#             sentvec.append(vec)
#         sentvec = np.mean(sentvec, 0)
#         embeddings.append(sentvec)
#
#     embeddings = np.vstack(embeddings)
#     return embeddings
