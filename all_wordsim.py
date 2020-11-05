import sys
import os
from tabulate import tabulate
from read_write import read_word_vectors
from ranking import *

def check_wordsim(word_vec_file, word_sim_dir):
    word_vecs = read_word_vectors(word_vec_file)
    # print('=================================================================================')
    # sys.stdout.write("%6s" %"Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho")
    # print('=================================================================================')
    table = []
    for i, filename in enumerate(os.listdir(word_sim_dir)):
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)
        for line in open(os.path.join(word_sim_dir, filename),'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in word_vecs and word2 in word_vecs:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
            else:
                not_found += 1
            total_size += 1
        table.append([filename, total_size, not_found, spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))])
    print(tabulate(table, headers=["Dataset", "Num Pairs", "Not Found", "Rho"]))
        # sys.stdout.write("%6s" % str(i+1), "%20s" % filename, "%15s" % str(total_size))
        # sys.stdout.write("%15s" % str(not_found))
        # sys.stdout.write("%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))
