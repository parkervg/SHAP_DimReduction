import json
import os
import glob
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_results(summary_dir: str=None, json_paths: list=None, label_bars=True):
    if summary_dir and json_paths:
        raise ValueError("Onle one of 'summary_dir', 'json_paths' can be specified.")
    if summary_dir: all_files = sorted(glob.glob("{}/*.json".format(summary_dir)), key=str.lower)
    elif json_paths: all_files = sorted(json_paths, key=str.lower)

    print(all_files)
    all_data = {}
    for filename in all_files:
        with open(filename) as f:
            data = json.load(f)
            all_data[filename] = data
    data = defaultdict(list)
    ordered_tasks = []
    classification_scores = defaultdict(list)
    for modelname, scores in all_data.items():
        modelname = os.path.splitext(os.path.basename(modelname))[0]
        for ix, (title, score) in enumerate(scores["classification_scores"].items()):
            data[modelname].append(score)
        if not ordered_tasks:
            ordered_tasks = [
                title for title in scores["classification_scores"]
            ]
    data["tasks"] = ordered_tasks
    df = pd.DataFrame(data=data)
    sns.set_theme(style="whitegrid")
    tidy = df.melt(id_vars="tasks").rename(columns=str.title)
    values = tidy["Value"].tolist()
    max_val = max(values) + 2
    if max_val > 100:
        max_val = 100
    min_val = min(values) - 10

    fig, ax = plt.subplots()
    fig.set_size_inches(len(all_files) * 2.85, len(all_files) * 1.5)

    ax = sns.barplot(
        ax=ax,
        data=tidy,
        x="Variable",
        y="Value",
        hue="Tasks",
        ci="sd",
        palette="dark",
        alpha=0.6,
    )
    if label_bars:
        values = range(len(ax.patches))
        for val, p in zip(values, ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.0, height + 1, height, ha="center")
    ax.set(xlabel="Word Vector", ylabel="Score")
    fig.suptitle("Scores Across Classification Tasks", fontsize=20)
    ax.set_ylim(min_val, max_val)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=14)
    plt.show()
