import itertools
import json
import io
import imp
import os
import uuid
import sys
import random
import math
import numpy as np
from sklearn.decomposition import PCA
from typing import Iterable, Dict, Any, Tuple, List, Sequence, Generator, Callable, Union
from tools.Blogger import Blogger
import tensorflow.compat.v1 as tf
from gensim.models.keyedvectors import KeyedVectors
import shap
from collections import defaultdict
from tabulate import tabulate
from tools.ranking import *
from tensorflow_fcwta.models import FullyConnectedWTA

EPSILON = 1e-6
RAND_STATE = 324
logger = Blogger()

# Set PATHs
PATH_TO_SENTEVAL = "./SentEval"
PATH_TO_DATA = "./SentEval/data"
WORD_SIM_DIR = "./data/word-sim"
# Interpretation of classes in TREC dataset
idx2tgt = {0: "ABBR", 1: "DESC", 2: "ENTY", 3: "HUM", 4: "LOC", 5: "NUM"}
task_explanations = {"SUBJ": "The label 1 refers to objective statements, and 0 refers to subjective statements",
                "CR": "A label of 1 is positive sentiment, and 0 is negative sentiment",
                "MR": "A label of 1 is positive sentiment, and 0 is negative sentiment",
                "MPQA": "A label of 1 is positive sentiment, and 0 is negative sentiment",
                "SST5": "Sentiment labels ranging from 0 (most negative) to 4 (most positive)",
                "SST2": "A label of 1 is positive sentiment, and 0 is negative sentiment"}

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST2", "SST5", "TREC", "MRPC"]
SIMILARITY_TASKS = ["SICKRelatedness", "STS12", "STS13", "STS14", "STS15", "STS16"]
"""
Increased scores on baseline tests:
    - Data leakage?
    - Or, cutting out the 'noisy' dimensions and allowing model to classify based on aspects relevant to task
TODO:
    - Offer algorithmic way of deciding optimal SHAP dimensions to take
    - Do analysis over variance explained per SHAP dimensions
"""


class WordEmbeddings:
    def __init__(
        self, vector_file: str, is_word2vec: bool = False, normalize_on_load: bool = False
    ):
        self.vector_file = vector_file if vector_file else "./embeds/glove.6B.300d.txt"
        # On normalization:
        # Levy et. al. 2015
        #   "Vectors are normalized to unit length before they are used for similarity calculation,
        #    making cosine similarity and dot-product equivalent.""
        self.normalize_on_load = normalize_on_load
        self.prev_components = np.empty((0, 0))
        self.function_log = []
        self.train_ngrams = []
        self.class_shaps = {}
        self.clf = None
        self.X_train = None
        if is_word2vec:
            self.load_word2vec_vectors()
        else:
            self.load_vectors()

    def load_word2vec_vectors(self):
        """
        Loader function for Word2Vec vectors.
        """
        logger.status_update("Loading vectors at {}...".format(self.vector_file))
        model = KeyedVectors.load_word2vec_format(
            "embeds/GoogleNews-vectors-negative300.bin", binary=True
        )
        self.embeds = model.vectors
        self.ordered_vocab = model.vocab.keys()
        # self.embeds = np.asarray(self.embeds) # Already a numpy array
        self.original_dim = self.embeds.shape[1]

    def load_vectors(self):
        """
        Loader function for Glove-formatted .txt vectors.
        """
        logger.status_update("Loading vectors at {}...".format(self.vector_file))
        self.ordered_vocab = []
        self.embeds = []
        with io.open(self.vector_file, "r", encoding="utf-8") as f:
            for line in f:
                word, vec = line.split(" ", 1)
                self.ordered_vocab.append(word)
                self.embeds.append(np.fromstring(vec, sep=" "))
                if self.normalize_on_load:
                    self.embeds[-1] /= math.sqrt((self.embeds[-1] ** 2).sum() + EPSILON)
        self.embeds = np.asarray(self.embeds)
        self.original_dim = self.embeds.shape[1]

    def pca_fit_transform(self, output_dims: int):
        """
        Fits and transforms embeds to output_dims with PCA.
        """
        self.function_log.append("pca_fit_transform")
        pca = PCA(n_components=output_dims, random_state=RAND_STATE)
        self.embeds = pca.fit_transform(self.embeds)
        self.prev_components = pca.components_

    def pca_fit(self):
        """
        Just fits PCA on embeds, doesn't change state of embeds.
        """
        self.function_log.append("pca_fit")
        pca = PCA(n_components=len(self.embeds[0]))
        pca.fit(self.embeds)
        self.prev_components = pca.components_

    def remove_top_components(self, k: int):
        """
        From a fitted PCA, removes the projection of the top k principle components in embeds.
        Lines 3-5 of Mu's post-processing algorithm.
        """
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
        """
        Subtracts mean from the embeds.
        """
        self.function_log.append("subract_mean")
        self.embeds = self.embeds - np.mean(self.embeds)

    def sparsify(self, checkpoint_path: str):
        """
        Calls a Tensoflow checkpoint of a fcwta training instance to sparsify embeddings
        """
        self.function_log.append("sparsify")
        flags_file = imp.load_source("flags.py", os.path.dirname(checkpoint_path) + "/flags.py")
        with tf.Session() as sess:
            fcwta = FullyConnectedWTA(
                self.original_dim,
                flags_file.FLAGS.batch_size,
                sparsity=flags_file.FLAGS.sparsity,
                hidden_units=flags_file.FLAGS.hidden_units,
                encode_layers=flags_file.FLAGS.num_layers,
                learning_rate=flags_file.FLAGS.learning_rate,
            )
            fcwta.saver.restore(sess, checkpoint_path)
            self.embeds = fcwta.encode(sess, self.embeds)
        tf.reset_default_graph()
        self._del_all_flags(flags_file.FLAGS)  # So next run won't raise error
        logger.status_update(f"New shape of embeds is {self.embeds.shape}")

    def shap_dim_reduction(self, task: str, k: int) -> List[int]:
        """
        Reduces embeds to only contain the top k dimensions identified by SHAP.
        """
        self.function_log.append("shap_dim_reduction")
        acc, self.clf, self.X_train, text = self.model_inference(task)
        self.make_train_ngrams(
            text[: self.X_train.shape[0]]
        )  # only sending text used from training
        logger.status_update(f"Original accuracy on task {task}: {acc}")
        dims = self.top_shap_dimensions(k=k)
        self.take_dims(dims)
        logger.status_update(f"New shape of embeds is {self.embeds.shape}")
        return dims

    def rand_dim_reduction(self, k: int, avoid_dims: List[int] = []) -> List[int]:
        """
        Used for testing purposes. Takes random selection from all of embedding dimensions
        """
        dims = random.sample([i for i in range(self.embeds.shape[1]) if i not in avoid_dims], k=k)
        logger.status_update(f"Randomly selected dimension indices {dims}")
        self.take_dims(dims)
        logger.status_update(f"New shape of embeds is {self.embeds.shape}")
        return dims

    def make_train_ngrams(self, train_text: str, n: int = 3):
        """
        Converts train_text to ngrams, flattens samples, and only keeps unique ngrams.
        """
        logger.status_update("Creating train ngrams...")
        ngrammed_text = [self.get_ngrams(sample, n=n) for sample in train_text]
        self.train_ngrams = set(tuple(gram) for gram in list(
            itertools.chain.from_iterable(ngrammed_text)
        ))  # Flatten list of lists

    @staticmethod
    def get_ngrams(text: List[str], n: int = 3) -> List[str]:
        """
        Gets all lowercased ngrams up to n, not including unigrams (single words)
        """
        text = [i.lower() for i in text]
        output = []
        for ix, word in enumerate(text):
            for i in range(1, n):
                to_add = text[ix : ix + i + 1]
                if not to_add in output:
                    output.append(to_add)
        return output

    def top_shap_dimensions(self, k: int) -> List[int]:
        """
        Averages over absolute shap values for each dimension, returning k top dimensions.
        """
        explainer = shap.LinearExplainer(self.clf, self.X_train, feature_dependence="independent")
        shap_values = explainer(self.X_train)
        logger.log(f"Classifier has {len(self.clf.classes_)} classes")
        if len(self.clf.classes_) == 2:
            vals = np.abs(shap_values.values).mean(0)
            # Each dimension index, sorted descending by sum of shap score
            sorted_dimensions = np.argsort(-vals, axis=0)
        else:
            vals = np.sum(np.abs(shap_values.values), axis=2).mean(0)
            sorted_dimensions = np.argsort(-vals, axis=0)
        return sorted_dimensions[:k]

    def shap_by_class(self, k: int = 10):
        """
        Identifies those dimensions with the highest average shap score for each predicted class.
        """
        logger.status_update("Finding top dimensions across classes...")
        explainer = shap.LinearExplainer(self.clf, self.X_train, feature_dependence="independent")
        shap_values = explainer(self.X_train)
        if len(self.clf.classes_) == 2:
            # For binary classification: negative shap implies push to class 0, positive to 1
            sorted_values = np.argsort(shap_values.values.mean(0), axis=0)
            class_zero_dims = sorted_values[:k]
            class_one_dims = sorted_values[-k:]
            self.class_shaps = {0: class_zero_dims, 1: class_one_dims}
        else:
            vals = np.sum(shap_values.values, axis=0)
            for label_ind in range(vals.shape[1]):
                scores = vals[:, label_ind]
                sorted_dimensions = np.argsort(-scores, axis=0)
                self.class_shaps[label_ind] = sorted_dimensions[:k]

    def top_ngrams_per_class(self, task: str, k: int = 10, n: int = 3) -> Dict[str, float]:
        """
        Logs the ngrams with the highest subspace score.
        """
        if not self.clf or self.X_train:
            _, self.clf, self.X_train, text = self.model_inference(task)
        if not self.class_shaps:
            self.shap_by_class()
        if not self.train_ngrams:
            self.make_train_ngrams(
                text[: self.X_train.shape[0]], n=n
            )  # only sending text used from training
        self.vector_dict = self.get_vector_dict()
        out = {}
        logger.status_update("Calculating subspace scores for ngrams...")

        if task in task_explanations:
            print()
            logger.yellow(task_explanations[task])
            print()
        for class_label, shap_dims in self.class_shaps.items():
            if task == "TREC":
                class_label = idx2tgt[class_label]
            subspace_scores = []
            out["class_label"] = {}
            out["class_label"]["dims"] = shap_dims
            # logger.status_update(f"SHAP dimensions for {class_label}:")
            # logger.log(shap_dims)
            for ngram in self.train_ngrams:
                subspace_scores.append((ngram, self.subspace_score(ngram, shap_dims)))
            logger.status_update(f"Top ngrams for class {class_label}:")
            for x in sorted(subspace_scores, key=lambda x: x[1], reverse=True)[:15]:
                print(" ".join(x[0]))
            print()
            out[class_label] = {}
            out[class_label]["ngrams"] = sorted(subspace_scores, key=lambda x: x[1], reverse=True)
        return out

    def subspace_score(self, text: Union[str, list], dims: List[int]) -> float:
        """
        As defined in Jang et al.
        Lowercases all words and returns subspace score.
        """
        if isinstance(text, str):
            v = self.vector_dict[text]
        else:  # Input is a list of words, average over all vectors
            v = []
            for word in text:
                v.append(self.vector_dict[word])
            v = np.mean(v, 0)
        if np.isinf(v).any():
            return 0.0
        slice = np.take(v, indices=dims)
        sum_slice = np.sum(slice, axis=0)
        return sum_slice / len(dims)

    def take_dims(self, dims: list):
        """
        Restricts self.embeds to only those dimensions whose indices are in dims.
        """
        self.embeds = np.take(self.embeds, indices=dims, axis=1)

    def model_inference(self, task: str):
        """
        Returns the classfier and X, text data used in SentEval task
        """
        # Set params for SentEval
        params_senteval = {"task_path": PATH_TO_DATA, "usepytorch": False, "kfold": 5}
        params_senteval["classifier"] = {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        }
        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare)
        results = se.eval(task)
        return results["acc"], results["classifier"], results["X_train"], results["text"]

    ############################################################################
    ####################### EVALUATION FUNCTIONS ###############################
    ############################################################################
    def get_vector_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns defaultdict of structure {word:vector}.
        Default is -inf for missing words
        """
        d = defaultdict(lambda: float('-inf'))
        for k, v in zip(self.ordered_vocab, self.embeds):
            d[k] = v
        return d

    @staticmethod
    def _del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = flags_dict.keys()
        for keys in keys_list:
            FLAGS.__delattr__(keys)

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
        words["<s>"] = 1e9 + 4
        words["</s>"] = 1e9 + 3
        words["<p>"] = 1e9 + 2

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
        logger.log(
            "Found {0} words with word vectors, out of \
            {1} words".format(
                len(word_vec), len(word2id)
            )
        )
        return word_vec

    @staticmethod
    def batcher(params, batch):
        batch = [sent if sent != [] else ["."] for sent in batch]
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

    # def similarity_tasks(self, save_summary=False, summary_file_name=None):
    #     self.summary["similarity_scores"] = {}
    #     word_vecs = {word: vector for word, vector in zip(self.ordered_vocab, self.embeds)}
    #     # Normalize for similarity tasks
    #     # Levy et. al. 2015
    #     #   "Vectors are normalized to unit length before they are used for similarity calculation,
    #     #    making cosine similarity and dot-product equivalent.""
    #     word_vecs = {
    #         word: vector / math.sqrt((vector ** 2).sum() + EPSILON)
    #         for word, vector in word_vecs.items()
    #     }
    #     table = []
    #     for i, filename in enumerate(os.listdir(WORD_SIM_DIR)):
    #         manual_dict, auto_dict = ({}, {})
    #         not_found, total_size = (0, 0)
    #         for line in open(os.path.join(WORD_SIM_DIR, filename), "r"):
    #             line = line.strip().lower()
    #             word1, word2, val = line.split()
    #             if word1 in word_vecs and word2 in word_vecs:
    #                 manual_dict[(word1, word2)] = float(val)
    #                 auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
    #             else:
    #                 not_found += 1
    #             total_size += 1
    #         rho = round(
    #             spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)) * 100,
    #             2,
    #         )
    #         self.summary["similarity_scores"][filename] = rho
    #         table.append([filename, total_size, not_found, rho])
    #     print(tabulate(table, headers=["Dataset", "Num Pairs", "Not Found", "Rho"]))

    def evaluate(
        self,
        senteval_tasks,
        save_summary=False,
        overwrite=False,
        summary_file_name=None,
        senteval_config={},
    ):
        """
        Runs SentEval classification tasks, and similarity tasks from Half-Size.
        """
        self.summary = {}
        self.run_senteval(
            senteval_tasks,
            save_summary=save_summary,
            summary_file_name=summary_file_name,
            senteval_config=senteval_config,
        )
        self.summary["original_dim"] = self.original_dim
        self.summary["final_dim"] = self.embeds.shape[1]
        self.summary["process"] = self.function_log
        self.summary["original_vectors"] = os.path.basename(self.vector_file)
        if save_summary:
            summary_file_name = summary_file_name if summary_file_name else str(uuid.uuid4())
            self.save_summary_json(summary_file_name, overwrite=overwrite)

    def run_senteval(self, tasks, save_summary=False, summary_file_name=None, senteval_config={}):
        if not isinstance(tasks, list):
            tasks = [tasks]
        # Set params for SentEval
        params_senteval = {
            "task_path": PATH_TO_DATA,
            "usepytorch": senteval_config.get("usepytorch")
            if senteval_config.get("usepytorch")
            else False,
            "kfold": senteval_config.get("kfold") if senteval_config.get("kfold") else 5,
        }
        params_senteval["classifier"] = {
            "nhid": senteval_config.get("nhid") if senteval_config.get("nhid") else 0,
            "optim": senteval_config.get("optim") if senteval_config.get("optim") else "rmsprop",
            "batch_size": senteval_config.get("batch_size")
            if senteval_config.get("batch_size")
            else 128,
            "tenacity": senteval_config.get("tenacity") if senteval_config.get("tenacity") else 3,
            "epoch_size": senteval_config.get("epoch_size")
            if senteval_config.get("epoch_size")
            else 2,
        }
        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare)
        self.summary["classification_scores"] = {}
        self.summary["similarity_scores"] = {}
        results = se.eval(tasks)
        for k in results:
            if k in CLASSIFICATION_TASKS:
                self.summary["classification_scores"][k] = results[k]["acc"]
                logger.status_update("{}: {}".format(k, results[k]["acc"]))
                print()
            elif k in SIMILARITY_TASKS:
                self.summary["similarity_scores"][k] = results[k]
                logger.status_update("{}: {}".format(k, results[k]["all"]["spearman"]["mean"]))
                print()

    def save_summary_json(self, summary_file_name, overwrite):
        if not os.path.isdir("summary"):
            os.mkdir("summary")
        if os.path.exists("summary/{}".format(summary_file_name)):
            if overwrite:
                logger.yellow("Existing summary file found, overwriting...")
                with open("summary/{}".format(summary_file_name), "w") as f:
                    json.dump(self.summary, f)
            else:
                logger.yellow("Existing summary file found, appending new output")
                with open("summary/{}".format(summary_file_name)) as f:
                    existing_data = json.load(f)
                existing_data = self.append_to_output(existing_data, "classification_scores")
                existing_data = self.append_to_output(existing_data, "similarity_scores")
                with open("summary/{}".format(summary_file_name), "w") as f:
                    json.dump(existing_data, f)
        else:
            with open("summary/{}".format(summary_file_name), "w") as f:
                json.dump(self.summary, f)
        logger.status_update("Summary saved to summary/{}".format(summary_file_name))

    def append_to_output(self, existing_data, section):
        for task, score in self.summary[section].items():
            if task not in existing_data[section]:
                existing_data[section][task] = score
            else:
                raise ValueError(f"Existing score already exists for task {task}")
        return existing_data

    def save_vectors(self, output_file):
        """
        Saves vectors to .txt file.
        """
        vector_size = self.embeds.shape[1]
        assert (len(self.ordered_vocab), vector_size) == self.embeds.shape
        with open(output_file, "w", encoding="utf-8") as out:
            for ix, word in enumerate(self.ordered_vocab):
                out.write("%s " % word)
                for t in self.embeds[ix]:
                    out.write("%f " % t)
                out.write("\n")
        logger.status_update("Vectors saved to {}".format(output_file))
