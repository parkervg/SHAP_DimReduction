"""
Run in Google Colab.
"""
from lib.ProcessEmbeddings import WordEmbeddings
from tools.Blogger import Blogger
from collections import defaultdict
logger = Blogger()
BINARY_CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA"]
MULTICLASS_CLASSIFICATION_TASKS = ["SST5", "TREC"]
CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST5", "TREC"]
SIMILARITY_TASKS = ['STS12', 'STS13', 'STS14']
ALL_TASKS = BINARY_CLASSIFICATION_TASKS + MULTICLASS_CLASSIFICATION_TASKS + SIMILARITY_TASKS
PRODUCTION_CONFIG = {"usepytorch": True, "kfold": 5, "nhid": 0, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 2}


def glove(output_dir, dims, senteval_config):
    summary_file_name=f"{output_dir}/glove_{dims}.json"
    WE = WordEmbeddings(vector_file=f"embeds/glove.6B.{dims}d.txt")
    # Default Glove
    WE.evaluate(tasks=CLASSIFICATION_TASKS,
                save_summary=True,
                summary_file_name=summary_file_name,
                overwrite_file=True,
                senteval_config=senteval_config)


def algo_n(WE, output_dir, dims, senteval_config):
    summary_file_name=f"{output_dir}/algo_{dims}.json"
    # PPE
    WE.subract_mean()
    WE.pca_fit()
    WE.remove_top_components(k=7)

    # PCA dim reduction
    WE.subract_mean()
    WE.pca_fit_transform(output_dims=dims)

    # PPE
    WE.subract_mean()
    WE.pca_fit()
    WE.remove_top_components(k=7)

    logger.status_update("Running SentEval tasks...")
    WE.evaluate(tasks=CLASSIFICATION_TASKS,
                save_summary=True,
                summary_file_name=summary_file_name,
                overwrite_file=True,
                senteval_config=senteval_config)

    WE.reset()
    assert WE.vectors.shape[1] == 300
    return WE

def shap_algo(WE, output_dir, dims, senteval_config):
    summary_file_name=f"{output_dir}/shap-algo_{dims}.json"
    for task in CLASSIFICATION_TASKS:
      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      # SHAP dim reduction
      WE.shap_dim_reduction(task=task, k=dims)

      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task,
                  save_summary=True,
                  summary_file_name=summary_file_name,
                  overwrite_task=True,
                  senteval_config=senteval_config)

      WE.reset()
      assert WE.vectors.shape[1] == 300

    return WE

def shap_ppe(WE, output_dir, dims, senteval_config):
    summary_file_name=f"{output_dir}/shap-ppe_{dims}.json"
    for task in CLASSIFICATION_TASKS:
      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      # SHAP dim reduction
      WE.shap_dim_reduction(task=task, k=dims)

      # # PPE
      # WE.subract_mean()
      # WE.pca_fit()
      # WE.remove_top_components(k=7)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task,
                  save_summary=True,
                  summary_file_name=summary_file_name,
                  overwrite_task=True,
                  senteval_config=senteval_config)

      WE.reset()
      assert WE.vectors.shape[1] == 300

    return WE

def shap_(WE, output_dir, dims, senteval_config):
    summary_file_name=f"{output_dir}/shap_{dims}.json"
    for task in CLASSIFICATION_TASKS:
      WE.shap_dim_reduction(task=task, k=dims)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task,
                  save_summary=True,
                  summary_file_name=summary_file_name,
                  overwrite_task=True,
                  senteval_config=senteval_config)
      WE.reset()
      assert WE.vectors.shape[1] == 300

    return WE

def evaluate_vectors(vector_file, output_dir, prototype_config=True):
    if output_dir[-1] == "/": output_dir = output_dir[:-1]
    if not prototype_config:
        senteval_config = PRODUCTION_CONFIG
    else:
        senteval_config = None
    # Standard Glove vectors first
    for dim in [50,
                100,
                #150,
                200]:
        glove(output_dir, dim, senteval_config)
    summary_file_name=f"{output_dir}/glove_300.json"
    WE = WordEmbeddings(vector_file=vector_file)
    WE.evaluate(tasks=CLASSIFICATION_TASKS, save_summary=True, summary_file_name=summary_file_name, overwrite_file=True, senteval_config=senteval_config)
    for dim in [50, 100, 150, 200]:
        WE = algo_n(WE, output_dir, dim, senteval_config)
        WE = shap_algo(WE, output_dir, dim, senteval_config)
        WE = shap_ppe(WE, output_dir, dim, senteval_config)
        WE = shap_(WE, output_dir, dim, senteval_config)
