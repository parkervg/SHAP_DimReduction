"""
Run in Google Colab.
"""
from lib.ProcessEmbeddings import WordEmbeddings
from tools.Blogger import Blogger
from lib.visualize_results import visualize_results
logger = Blogger()
BINARY_CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA"]
MULTICLASS_CLASSIFICATION_TASKS = ["SST5", "TREC"]
CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST5", "TREC"]
SIMILARITY_TASKS = ['STS12', 'STS13', 'STS14']
ALL_TASKS = BINARY_CLASSIFICATION_TASKS + MULTICLASS_CLASSIFICATION_TASKS + SIMILARITY_TASKS

def evaluate_vectors(vector_file, output_dir):
    if output_dir[-1] == "/": output_dir = output_dir[:-1]
    WE = WordEmbeddings(vector_file=vector_file)

    """
    Default Glove
    300
    """
    summary_file_name=f"{output_dir}/glove.json"
    WE.evaluate(tasks=CLASSIFICATION_TASKS, save_summary=True, summary_file_name=summary_file_name, overwrite_file=True)

    """
    Algo-N
    150
    """
    summary_file_name=f"{output_dir}/algo-n.json"
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

    logger.status_update("Running SentEval tasks...")
    WE.evaluate(tasks=CLASSIFICATION_TASKS, save_summary=True, summary_file_name=summary_file_name, overwrite_file=True)

    WE.reset()
    assert WE.embeds.shape[1] == 300

    """
    Shap-Algo with Glove
    150
    """
    summary_file_name=f"{output_dir}/shap-algo_150.json"
    for task in CLASSIFICATION_TASKS:
      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      # SHAP dim reduction
      WE.shap_dim_reduction(task=task, k=150)

      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task, save_summary=True, summary_file_name=summary_file_name, overwrite_task=True)

    WE.reset()
    assert WE.embeds.shape[1] == 300

    """
    Shap-Algo with Glove
    50
    """
    summary_file_name=f"{output_dir}/shap-algo_50.json"
    for task in CLASSIFICATION_TASKS:
      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      # SHAP dim reduction
      WE.shap_dim_reduction(task=task, k=50)

      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task, save_summary=True, summary_file_name=summary_file_name, overwrite_task=True)

    WE.reset()
    assert WE.embeds.shape[1] == 300


    """
    Shap-PPE with Glove
    50
    """
    summary_file_name=f"{output_dir}/shap-ppe_50.json"
    for task in CLASSIFICATION_TASKS:
      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      # SHAP dim reduction
      WE.shap_dim_reduction(task=task, k=50)

      # # PPE
      # WE.subract_mean()
      # WE.pca_fit()
      # WE.remove_top_components(k=7)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task, save_summary=True, summary_file_name="SHAP/shap-ppe_50.json", overwrite_task=True)

    WE.reset()
    assert WE.embeds.shape[1] == 300

    """
    Shap-PPE with Glove
    150
    """
    summary_file_name=f"{output_dir}/shap-ppe_150.json"
    for task in CLASSIFICATION_TASKS:
      # PPE
      WE.subract_mean()
      WE.pca_fit()
      WE.remove_top_components(k=7)

      # SHAP dim reduction
      WE.shap_dim_reduction(task=task, k=150)

      # # PPE
      # WE.subract_mean()
      # WE.pca_fit()
      # WE.remove_top_components(k=7)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task, save_summary=True, summary_file_name=summary_file_name, overwrite_task=True)

    WE.reset()
    assert WE.embeds.shape[1] == 300

    """
    Shap on Glove
    150
    """
    summary_file_name=f"{output_dir}/shap_150.json"
    for task in CLASSIFICATION_TASKS:
      WE.shap_dim_reduction(task=task, k=150)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task, save_summary=True, summary_file_name=summary_file_name, overwrite_task=True)
      WE.reset()
      assert WE.embeds.shape[1] == 300


    """
    Shap on Glove
    50
    """
    summary_file_name=f"{output_dir}/shap_50.json"
    for task in CLASSIFICATION_TASKS:
      WE.shap_dim_reduction(task=task, k=50)

      logger.status_update("Running SentEval tasks...")
      WE.evaluate(tasks=task, save_summary=True, summary_file_name=summary_file_name, overwrite_task=True)
      WE.reset()
      assert WE.embeds.shape[1] == 300

    WE.reset()
    assert WE.embeds.shape[1] == 300
