from lib.ProcessEmbeddings import WordEmbeddings
import argparse


def get_top_shap_grams(task, k, n):
    WE = WordEmbeddings(vector_file='embeds/glove.6B.300d.txt', is_word2vec=False)
    out = WE.top_ngrams_per_class(task=task, k=k, n=n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Grabs top ngrams per class based on SHAP dimensions'
    )
    parser.add_argument('task', help="The senteval classification task to test on. One of: 'MR', 'CR', 'SUBJ', 'MPQA', 'STS', 'SST', 'TREC', 'MRPC'.")
    parser.add_argument('-k', type=int, default=30, help="The number of dimensions to take into consideration when calculating the subspace score.")
    parser.add_argument('--ngram_size', type=int, default=3, help="The size of the ngram window.")
    args = parser.parse_args()
    get_top_shap_grams(args.task, args.k, args.ngram_size)
