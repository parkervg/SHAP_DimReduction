from lib.ProcessEmbeddings import WordEmbeddings
from tools.ranking import *
import shap
import numpy as np
from scipy import spatial

WE = WordEmbeddings(vector_file="embeds/glove.6B.300d.txt")
GLOVE = {word: vector for word, vector in zip(WE.ordered_vocab, WE.embeds)}
GLOVE = {
    word: vector / math.sqrt((vector ** 2).sum() + EPSILON)
    for word, vector in GLOVE.items()
}
WE.sparsify("ckpt/glove3000/ckpt-8000")
SPARSE_GLOVE = {word: vector for word, vector in zip(WE.ordered_vocab, WE.embeds)}
SPARSE_GLOVE = {
    word: vector / math.sqrt((vector ** 2).sum() + EPSILON)
    for word, vector in SPARSE_GLOVE.items()
}

cosine_sim(SPARSE_GLOVE["agent"], SPARSE_GLOVE["spy"])
cosine_sim(GLOVE["agent"], GLOVE["spy"])


cosine_sim(SPARSE_GLOVE["blue"], SPARSE_GLOVE["republic"])
cosine_sim(GLOVE["blue"], GLOVE["republic"])


#########################################################
# https://github.com/slundberg/shap/blob/master/notebooks/linear_explainer/Sentiment%20Analysis%20with%20Logistic%20Regression.ipynb
from custom_shap import shap
import shap
from lib.ProcessEmbeddings import WordEmbeddings
import numpy as np
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from tools.ranking import cosine_sim
from gensim.models.keyedvectors import KeyedVectors
"""
For binary classification:
    - SHAP values give positive scores to those variables pushing prediction to 1, negative to 0
    - Bar plot just averages the mean SHAP score across all prediction instances
        mean_features = np.mean(np.abs(shap_values.values), axis=0)
        - The max() features here are the most discriminatory dimensions (pushes classification towards a decision the most)
    - These features aren't the most discriminatory, actually pretty ambiguous
"""
WE = WordEmbeddings(vector_file="embeds/glove_algo150.txt", is_word2vec=False)
WE.evaluate(senteval_tasks=["CR"], save_summary=True, summary_file_name="glove.json")




WE = WordEmbeddings(vector_file="embeds/glove_algo150.txt", is_word2vec=False)
WE.shap_dim_reduction("CR")
WE.evaluate(senteval_tasks=["CR"])

WE = WordEmbeddings(vector_file="embeds/glove_algo150.txt", is_word2vec=False)
WE.pca_fit_transform(output_dims=10)
WE.evaluate(senteval_tasks=["CR"])



# _, clf, X, Y = WE.model_inference("TREC")

_, clf, X, Y = WE.model_inference("CR")
explainer = shap.Explainer(clf, X)
shap_values = explainer(X)
vals = np.abs(shap_values)
vals = np.abs(shap_values.values).mean(0)




dim1, dim2 = top_dimensions_binary(clf, X, by_class=True)
dims = top_dimensions_binary(clf, X, by_class=False)
word_vectors = {k:v for k, v in zip(WE.ordered_vocab, WE.embeds)}

all_scores = []
for word in WE.ordered_vocab:
    score = subspace_score(word, [276, 14])
    if score > 0:
        all_scores.append((word, score))

sorted(all_scores, key = lambda x: x[1], reverse=True)[:20]
test_ind = [276, 14]

shap.plots.beeswarm(shap_values, max_display=100)
shap.plots.bar(shap_values)

mean_features = np.argsort(np.mean(np.abs(values), axis=0))
np.argsort(np.mean(np.abs(shap_values.values), axis=0))


dims = top_dimensions_binary(clf, X, k = 10)
embeds = np.take(WE.embeds, indices=dims, axis=1)
embeds.shape
WE.embeds.shape


def subspace_score(word: str, dims: list) -> float:
  """
  As defined in Jang et al.
  """
  v = word_vectors[word]
  slice = np.take(v, indices=dims)
  sum_slice = np.sum(slice, axis=0)
  return (sum_slice / len(dims))


def top_dimensions_binary(clf, X, k = 10):
    if len(clf.classes_) != 2:
        raise ValueError(f"Classifier is not binary, predicting on {len(clf.classes_)} classes")
    explainer = shap.Explainer(clf, X)
    shap_values = explainer(X)
    vals = np.abs(shap_values.values).mean(0)
    # Each dimension index, sorted descending-first by sum of shap score
    sorted_dimensions = np.argsort(-vals, axis=0)
    return sorted_dimensions[:k]




cohorts = {"": shap_values}
cohort_labels = list(cohorts.keys())
cohort_exps = list(cohorts.values())
for i in range(len(cohort_exps)):
    if len(cohort_exps[i].shape) == 2:
        cohort_exps[i] = cohort_exps[i].abs.mean(0)
features = cohort_exps[0].data
feature_names = cohort_exps[0].feature_names

partition_tree = getattr(cohort_exps[0], "clustering", None)







import pickle

with open("X.pkl", "rb") as f:
    X = pickle.load(f)
with open("Y.pkl", "rb") as f:
    Y = pickle.load(f)
with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)
import numpy as np
