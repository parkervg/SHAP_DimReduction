from sklearn.decomposition import PCA
import numpy as np
from tools.Blogger import Blogger
import tensorflow.compat.v1 as tf
from tensorflow_fcwta.models import FullyConnectedWTA
from gensim.models.keyedvectors import KeyedVectors
import shap
import json
import io
import imp
import os
import uuid
import sys
import random
import math
from tabulate import tabulate
from tools.ranking import *

EPSILON = 1e-6
RAND_STATE = 324
logger = Blogger()

# Set PATHs
PATH_TO_SENTEVAL = "./SentEval"
PATH_TO_DATA = "./SentEval/data"
WORD_SIM_DIR = "./data/word-sim"

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA", "STS", "SST", "TREC", "MRPC"]
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
    def __init__(self, vector_file, is_word2vec=False, normalize_on_load=False):
        self.vector_file = vector_file if vector_file else "./embeds/glove.6B.300d.txt"
        # On normalization:
        # Levy et. al. 2015
        #   "Vectors are normalized to unit length before they are used for similarity calculation,
        #    making cosine similarity and dot-product equivalent.""
        self.normalize_on_load = normalize_on_load
        self.prev_components = np.empty((0, 0))
        self.function_log = []
        self.train_vocab = []
        if is_word2vec:
            self.load_word2vec_vectors()
        else:
            self.load_vectors()

    def load_word2vec_vectors(self):
        logger.status_update("Loading vectors at {}...".format(self.vector_file))
        model = KeyedVectors.load_word2vec_format(
            "embeds/GoogleNews-vectors-negative300.bin", binary=True
        )
        self.embeds = model.vectors
        self.ordered_vocab = model.vocab.keys()
        # self.embeds = np.asarray(self.embeds) # Already a numpy array
        self.original_dim = self.embeds.shape[1]

    def load_vectors(self):
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
            raise ValueError(
                "No value found for prev_components. Did you call pca_fit_transform?"
            )
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
        flags_file = imp.load_source(
            "flags.py", os.path.dirname(checkpoint_path) + "/flags.py"
        )
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

    def shap_dim_reduction(self, task, k):
        self.function_log.append("shap_dim_reduction")
        acc, clf, X_train, text = self.model_inference(task)
        self.make_train_vocab(text[:X_train.shape[0]]) # only sending text used from training
        logger.status_update(f"Original accuracy on task {task}: {acc}")
        dims = self.top_shap_dimensions(clf, X_train, k=k)
        self.take_dims(dims)
        logger.status_update(f"New shape of embeds is {self.embeds.shape}")
        return dims

    def rand_dim_reduction(self, k, avoid_dims=[]):
        """
        Used for testing purposes. Takes random selection from all of embedding dimensions
        """
        dims = random.sample([i for i in range(self.embeds.shape[1]) if i not in avoid_dims], k=k)
        logger.status_update(f"Randomly selected dimension indices {dims}")
        self.take_dims(dims)
        logger.status_update(f"New shape of embeds is {self.embeds.shape}")
        return dims

    def make_train_vocab(self, train_text):
        """
        Flattens train_text and only keeps unique words.
        """
        self.train_text = set([word for sample in train_text for word in sample])


    @staticmethod
    def top_shap_dimensions(clf, X_train, k):
        explainer = shap.LinearExplainer(clf, X_train, feature_dependence="independent")
        shap_values = explainer(X_train)
        logger.log(f"Classifier has {len(clf.classes_)} classes")
        if len(clf.classes_) == 2:
            vals = np.abs(shap_values.values).mean(0)
            # Each dimension index, sorted descending by sum of shap score
            sorted_dimensions = np.argsort(-vals, axis=0)
        else:
            vals = np.sum(np.abs(shap_values.values), axis=2).mean(0)
            sorted_dimensions = np.argsort(-vals, axis=0)
        return sorted_dimensions[:k]

    @staticmethod
    def subspace_score(word: str, dims: list) -> float:
        """
        As defined in Jang et al.
        """
        v = word_vectors[word]
        slice = np.take(v, indices=dims)
        sum_slice = np.sum(slice, axis=0)
        return (sum_slice / len(dims))


    # @staticmethod
    # def top_shap_dimensions_multi(clf, X, k):
    #     if len(clf.classes_) == 2:
    #         raise ValueError(
    #             f"Classifier is not multiclass, predicting on {len(clf.classes_)} classes"
    #         )
    #     explainer = shap.Explainer(clf, X)
    #     shap_values = explainer(X)
    #     vals = np.sum(shap_values.values, axis=0)
    #     dim_per_label = int(k / len(clf.classes_))
    #     logger.log(f"Selecting {dim_per_label} dimensions per label...")
    #     top_dims = {}
    #     used_dims = set()
    #     for label_ind in range(vals.shape[1]):
    #         scores = vals[:, label_ind]
    #         sorted_dimensions = np.argsort(-scores, axis=0)
    #         top_dims[label_ind] = [i for i in sorted_dimensions if i not in used_dims][:dim_per_label]
    #         for dim in top_dims[label_ind]:
    #             used_dims.add(dim)
    #     dims = [item for sublist in top_dims.values() for item in sublist]
    #     return dims

    def take_dims(self, dims):
        """
        Restricts self.embeds to only those dimensions whose indices are in dims.
        """
        self.embeds = np.take(self.embeds, indices=dims, axis=1)

    def model_inference(self, task):
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
    def get_vector_dict(self):
        """
        Returns traditional word_vector dict, of structure {word:vector}
        """
        return {k:v for k, v in zip(self.ordered_vocab, self.embeds)}


    @staticmethod
    def _del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
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

    def similarity_tasks(self, save_summary=False, summary_file_name=None):
        self.summary["similarity_scores"] = {}
        word_vecs = {
            word: vector for word, vector in zip(self.ordered_vocab, self.embeds)
        }
        # Normalize for similarity tasks
        # Levy et. al. 2015
        #   "Vectors are normalized to unit length before they are used for similarity calculation,
        #    making cosine similarity and dot-product equivalent.""
        word_vecs = {
            word: vector / math.sqrt((vector ** 2).sum() + EPSILON)
            for word, vector in word_vecs.items()
        }
        table = []
        for i, filename in enumerate(os.listdir(WORD_SIM_DIR)):
            manual_dict, auto_dict = ({}, {})
            not_found, total_size = (0, 0)
            for line in open(os.path.join(WORD_SIM_DIR, filename), "r"):
                line = line.strip().lower()
                word1, word2, val = line.split()
                if word1 in word_vecs and word2 in word_vecs:
                    manual_dict[(word1, word2)] = float(val)
                    auto_dict[(word1, word2)] = cosine_sim(
                        word_vecs[word1], word_vecs[word2]
                    )
                else:
                    not_found += 1
                total_size += 1
            rho = round(
                spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)) * 100,
                2,
            )
            self.summary["similarity_scores"][filename] = rho
            table.append([filename, total_size, not_found, rho])
        print(tabulate(table, headers=["Dataset", "Num Pairs", "Not Found", "Rho"]))

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
            summary_file_name = (
                summary_file_name if summary_file_name else str(uuid.uuid4())
            )
            self.save_summary_json(summary_file_name, overwrite=overwrite)

    def run_senteval(
        self, tasks, save_summary=False, summary_file_name=None, senteval_config={}
    ):
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
            "optim": senteval_config.get("optim")
            if senteval_config.get("optim")
            else "rmsprop",
            "batch_size": senteval_config.get("batch_size")
            if senteval_config.get("batch_size")
            else 128,
            "tenacity": senteval_config.get("tenacity")
            if senteval_config.get("tenacity")
            else 3,
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
                logger.status_update(
                    "{}: {}".format(k, results[k]["all"]["spearman"]["mean"])
                )
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
