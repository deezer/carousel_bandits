import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style(style='darkgrid')


# Plots the evolution of expected cumulative regrets curves,
# for all tested policies and over all rounds
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="results.json", required=False,
                        help="path to data")

    args = parser.parse_args()

    with open(args.data_path, 'r') as fp:
        cumulative_regrets = json.load(fp)

    for k,v in cumulative_regrets.items():
        sns.lineplot(data = np.array(v), label=k)
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.show()