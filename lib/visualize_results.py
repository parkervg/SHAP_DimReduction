import json
import os
import glob
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.interpolate import interp1d

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_bar(summary_dir: str=None, json_paths: list=None, label_bars=True):
    """
    Creates seaborn grouped bar chart of scores on senteval classification tasks.
    """
    if summary_dir and json_paths:
        raise ValueError("Only one of 'summary_dir', 'json_paths' can be specified.")
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

def create_scatter(task, summary_dir: str=None, json_paths: list=None):
    if summary_dir and json_paths:
        raise ValueError("Only one of 'summary_dir', 'json_paths' can be specified.")
    if summary_dir: all_files = sorted(glob.glob("{}/*.json".format(summary_dir)), key=str.lower)
    elif json_paths: all_files = sorted(json_paths, key=str.lower)

    algo_n = sorted([f for f in all_files if re.search(r'algo-n_.*?(?=\.json)', f)], key=natural_keys)
    shap_algo = sorted([f for f in all_files if re.search(r'shap-algo_.*?(?=\.json)', f)], key=natural_keys)
    shap_ppe = sorted([f for f in all_files if re.search(r'shap-ppe_.*?(?=\.json)', f)], key=natural_keys)
    shap_ = sorted([f for f in all_files if re.search(r'shap_.*?(?=\.json)', f)], key=natural_keys)

    data_dict = defaultdict(list)
    df = pd.DataFrame(columns=['vector_name', 'accuracy', 'dimensions'])
    for group in [algo_n, shap_algo, shap_ppe, shap_]:
        for filename in group:
            with open(filename) as f:
                dims = int(re.search(r'\d\d(\d)?(?=\.json)', filename).group())
                data = json.load(f)
                vector_name = re.sub(r'_\d\d(\d)?$', '', os.path.splitext(os.path.basename(filename))[0])
                df.loc[-1] = [vector_name.upper(), data['classification_scores'][task], dims]
                df.index = df.index + 1
                df.sort_index()
                data_dict[re.sub(r'_\d\d(\d)?$', '', os.path.splitext(os.path.basename(filename))[0])].append(data['classification_scores'][task])


    x = [50, 100, 150, 200]
    markers = ['^', 'o', 'D', '*']
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    sns.set_palette("husl")
    sns.despine(ax=ax, left=True, top=True, bottom=True, right=True)
    plt.grid(axis='x', b=None)
    ax.set_xticks(ticks=x)
    sp = sns.scatterplot(data=df, x="dimensions", y="accuracy", hue="vector_name", style="vector_name", s=90)
    sp.set_xlabel("Dimensions", fontsize=12)
    sp.set_ylabel("Accuracy", fontsize=12)
    for ix, (vector_name, scores) in enumerate(data_dict.items()):
        x_new = np.linspace(min(x), max(x), 500)
        f = interp1d(x, scores, kind="quadratic")
        y_smooth=f(x_new)
        sns.lineplot(x_new, y_smooth)
    plt.legend(
               frameon=False,
               bbox_to_anchor=(0.5, 1.05),
               loc='upper center',
               borderaxespad=0.0,
               fontsize=14,
               ncol=4
               )
    plt.rcParams["font.family"] = "Times"
    fig.suptitle(f"{task} Scores", fontsize=20, fontweight="bold")
