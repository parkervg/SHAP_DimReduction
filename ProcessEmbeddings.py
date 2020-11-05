from sklearn.decomposition import PCA
import numpy as np
from tools.Blogger import Blogger
from all_wordsim import check_wordsim
import io
EPSILON = 1e-6
RAND_STATE = 324
logger = Blogger()
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
        pca = PCA(n_components=output_dims, random_state=RAND_STATE)
        self.embeds = pca.fit_transform(self.embeds)
        self.prev_components = pca.components_

    def pca_fit(self):
        pca = PCA(n_components=len(self.embeds[0]))
        pca.fit(self.embeds)
        self.prev_components = pca.components_

    def remove_top_components(self, k):
        if self.prev_components.size == 0:
            raise ValueError("No value found for prev_components. Did you call pca_fit_transform?")
        z = []
        for ix, x in enumerate(self.embeds):
            for u in self.prev_components[0:k]:
                x = x - np.dot(u.transpose(), x) * u
            z.append(x)
        self.embeds = np.asarray(z)

    def subract_mean(self):
        self.embeds = self.embeds - np.mean(self.embeds)

    def save_vectors(self, output_file):
        vector_size = self.embeds.shape[1]
        assert (len(self.ordered_vocab), vector_size)) == self.embeds.shape
        with open(output_file, "w", encoding="utf-8") as out:
            for ix, word in enumerate(self.ordered_vocab):
                out.write("%s " % word)
                for t in self.embeds[ix]:
                    out.write("%f " % t)
                out.write("\n")




WE = WordEmbeddings(vector_file="embeds/glove.6B.300d.txt")
# PPE
WE.subract_mean()
WE.pca_fit()
WE.remove_top_components(k=7)

# PCA dim reduction
WE.subract_mean()
WE.pca_fit_transform(output_dims=150)

# PPE
WE.subract_mean()
WE.pca_fit()
WE.remove_top_components(k=7)
WE.save_vectors(output_file="embeds/glove_algo150.txt")

check_wordsim("embeds/glove_algo.txt", "data/word-sim/")
check_wordsim("embeds/glove_algo2.txt", "data/word-sim/")


output_file = "glove_algo.txt"
vector_size = WE.embeds.shape[1]
assert (len(WE.ordered_vocab), vector_size) == WE.embeds.shape
with open(output_file, "w", encoding="utf-8") as out:
    for ix, word in enumerate(WE.ordered_vocab):
        out.write("%s " % word)
        for t in WE.embeds[ix]:
            out.write("%f " % t)
        out.write("\n")


WE = WordEmbeddings(vector_file="embeds/glove_algo.txt")
[WE.embeds[i].shape for i in range(len(WE.embeds)) if WE.embeds[i].shape != 150]
