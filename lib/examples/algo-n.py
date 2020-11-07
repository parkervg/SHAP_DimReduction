from lib.ProcessEmbeddings import WordEmbeddings
from tools.Blogger import Blogger
import os

logger = Blogger()
CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST5", "TREC", "MRPC"]
SIMILARITY_TASKS = ["SICKRelatedness", "STS12", "STS13", "STS14", "STS15", "STS16"]
ALL_TASKS = CLASSIFICATION_TASKS + SIMILARITY_TASKS

if __name__ == "__main__":
    if not os.path.exists("embeds/glove_algo150.txt"):
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
        WE.save_vectors("embeds/glove_algo150.txt")
        logger.status_update("Running SentEval tasks...")

        WE.evaluate(
            senteval_tasks=ALL_TASKS, save_summary=True, summary_file_name="algo-n.json"
        )
    else:
        WE = WordEmbeddings(vector_file="embeds/glove_algo150.txt")
        logger.status_update("Running SentEval tasks...")
        WE.evaluate(
            senteval_tasks=ALL_TASKS, save_summary=True, summary_file_name="algo-n.json"
        )
