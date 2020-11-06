from lib.ProcessEmbeddings import WordEmbeddings
from tools.Blogger import Blogger
import os
logger = Blogger()
CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA", "STS", "SST", "TREC", "SICK", "MRPC"]

if __name__ == "__main__":
    WE = WordEmbeddings(vector_file="embeds/glove.6B.300d.txt")
    WE.sparsify("ckpt/glove3000/ckpt-8000")
    WE.subract_mean()
    logger.status_update("Running SentEval tasks...")
    WE.SentEval(tasks=CLASSIFICATION_TASKS, save_summary=True, summary_file_name="glove_wta_3000.json")
