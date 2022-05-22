import numpy as np
import warnings
from scipy.stats import ttest_rel
from tabulate import tabulate

from unigram_tagger import unigram_standalone, unigram_with_regexp
from brill_tagger import start_postag_brill_tagger

def main():
    warnings.filterwarnings('ignore', '.*The least populated*', )
    scores = np.zeros((3, 10))
    scores[0], _ = unigram_standalone()
    scores[1], _ = unigram_with_regexp()
    scores[2], _ = start_postag_brill_tagger()
    alfa = .05
    print(scores)
    print(scores.mean(axis=1))
    t_statistic = np.zeros((scores.shape[0], scores.shape[0]))
    p_value = np.zeros((scores.shape[0], scores.shape[0]))

    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

    headers = ["UNS", "UNR", "BRI"]
    names_column = np.array([["UNS"], ["UNR"], ["BRI"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
    advantage = np.zeros((scores.shape[0], scores.shape[0]))
    advantage[t_statistic > 0] = 1
    significance = np.zeros((scores.shape[0], scores.shape[0]))
    significance[p_value <= alfa] = 1
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)

if __name__ == '__main__':
    main()
