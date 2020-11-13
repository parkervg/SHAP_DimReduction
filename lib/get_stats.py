import json
import os
import glob
from collections import defaultdict
import pandas as pd
import re
import numpy as np
import statistics
from tools.analysis_tools import *
CLASSIFICATION_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST5", "TREC"]

def get_avg_ranks(summary_dir: str=None, json_paths: list=None) -> dict:
    """
    Returns dict of average rank across all classification tasks for each vector.
    """
    if summary_dir and json_paths:
        raise ValueError("Only one of 'summary_dir', 'json_paths' can be specified.")
    if summary_dir: all_files = sorted(glob.glob("{}/*.json".format(summary_dir)), key=str.lower)
    elif json_paths: all_files = sorted(json_paths, key=str.lower)

    all_data = {}
    for filename in all_files:
        with open(filename) as f:
            data = json.load(f)
            all_data[filename] = data
        data = defaultdict(lambda: defaultdict(list))
        ordered_tasks = []
        classification_scores = defaultdict(list)
        for name, scores in all_data.items():
            if name != 'summary/SHAP/glove.json':
                vectorname = re.sub(r'_\d\d(\d)?$', '', os.path.splitext(os.path.basename(name))[0])
                dims = int(re.search(r'\d\d(\d)?', os.path.splitext(os.path.basename(name))[0]).group())
                for ix, (task, score) in enumerate(scores["classification_scores"].items()):
                    data[task][dims].append((vectorname, score))

    all_vectors = set([IS_VECTOR.search(i).group() for i in all_files if not i.endswith("glove.json")])

    vector_places = defaultdict(list)
    for task, dims in data.items():
        for dim, results in dims.items():
            if dim != 300: # Only glove uses this
                results = sorted(results, key=lambda x: x[1], reverse=True)
                for vector in all_vectors:
                    try:
                        vector_places[vector].append([i[0] for i in results].index(vector) + 1)
                    except ValueError:
                        assert vector == "glove"
    avg_ranks = {}
    for vector, ranks in vector_places.items():
        avg_ranks[vector] = round(statistics.mean(ranks), 2)

    return avg_ranks


def get_score_table(summary_dir: str=None, json_paths: list=None):
    if summary_dir and json_paths:
        raise ValueError("Only one of 'summary_dir', 'json_paths' can be specified.")
    if summary_dir: all_files = sorted(glob.glob("{}/*.json".format(summary_dir)), key=str.lower)
    elif json_paths: all_files = sorted(json_paths, key=str.lower)
    df = pd.DataFrame(columns= ["Model"] + CLASSIFICATION_TASKS)
    for filename in all_files:
        with open(filename) as f:
            data = json.load(f)
            vector_name = os.path.splitext(os.path.basename(filename))[0]
            if vector_name.startswith("shap") or vector_name=="glove_300":
                row = [vector_name.upper()+"D"] + [data["classification_scores"][task] for task in CLASSIFICATION_TASKS]
                df = insert_row(df, row)
    return df



if __name__ == "__main__":
    get_avg_ranks(summary_dir="summary/SHAP/production")
    get_score_table(summary_dir="summary/SHAP/production")




### Averaging Random Data Results
import statistics
from collections import defaultdict
import pickle
with open("summary/rand_data.pkl", "rb") as f:
    rand_dims = pickle.load(f)
mean_scores = defaultdict(lambda defaultdict(float))
for dim, task in rand_dims.items():
    for task_name, scores in task.items():
        mean_score = statistics.mean(scores)
    mean_scores[dim][task_name] = mean_score
mean_scores[50]["task"]





### Analyzing specific examples
from lib.ProcessEmbeddings import WordEmbeddings
WE = WordEmbeddings(vector_file="embeds/glove.6B.300d.txt", normalize_on_load=True)
out1 = WE.analyze_sentence("CR", "and supply those stupid white headphones .")
out2 = WE.analyze_sentence("SUBJ", "The movie is about children who learn to fly. It is honestly quite ridiculous.", k=50)
