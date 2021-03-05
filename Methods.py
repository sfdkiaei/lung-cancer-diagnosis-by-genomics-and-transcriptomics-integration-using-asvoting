from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3
from matplotlib_venn_wordcloud import venn2_wordcloud
from Analysis import Analysis


class StatisticalTest:
    def __init__(self):
        pass

    def tTest(self, a, b):
        statistic, pvalue = stats.ttest_ind(a, b, equal_var=False)
        return pvalue


class Fusion:
    def __init__(self, analyzer: Analysis = None):
        self.analyzer = analyzer

    def owa(self, train_data, train_label, test_data, num_args, lr=0.9, epoch_num=150):
        # Initialize
        landa = np.random.rand(num_args)
        # landa = np.zeros(num_args)
        w = np.ones(num_args) * (1.0 / num_args)
        d_estimate = np.sum([np.exp(landa[i]) / np.sum(np.exp(landa)) for i in range(num_args)])
        # d_estimate = 0

        # Train
        for epoch in range(epoch_num):
            lr /= 1.001
            for idx, sample in enumerate(train_data):
                b = np.sort(sample)[::-1]
                diff = w * (b - d_estimate) * (d_estimate - train_label[idx])
                landa -= lr * diff
                w = [np.exp(landa[i]) / np.sum(np.exp(landa)) if np.exp(landa[i]) / np.sum(np.exp(landa)) > 1e-5 else 0
                     for
                     i in range(num_args)]
                # print(w)
                d_estimate = np.sum(b * w)

        # print('\nLambdas:', np.round(landa, 2))
        # print('Weights:', np.round(w, 2))
        # print('\nSample\t\tAggregated Value\tEstimated Value')
        # print('-----------------------------------------------')
        # for idx, sample in enumerate(train_data):
        #     print('Sample', (idx + 1), '\t\t', train_label[idx], '\t\t\t', np.round(np.sum(np.sort(train_data[idx])[::-1] * w), 2))
        # Prediction
        result = []
        for idx, sample in enumerate(test_data):
            b = np.sort(sample)[::-1]
            d_estimate = np.sum(b * w)
            result.append(d_estimate)
        return result

    def maxVoting(self):
        if self.analyzer is not None:
            result = []
            length = 0
            for i in self.analyzer.getPredictions():
                if i is not None:
                    length = len(i)
                    break
            if length == 0:
                raise ValueError('All classifiers predictions in Analysis class are None')
            for idx in range(length):
                result.append(np.max(self.analyzer.getPredictions()[:, idx]))
        else:
            raise AttributeError('Object from Analysis class should pass in the constructor')
        return result


class Visualization:
    def __init__(self):
        pass

    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", center=None, **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = sns.heatmap(data, center=center, square=False,
                         cbar_kws={"orientation": "horizontal", 'pad': 0.05, 'aspect': 50, 'label': cbarlabel})

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels, rotation=0)

        ax.set_xlabel('Housekeeping Genes')
        ax.set_ylabel('Driver Genes')

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        # ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # return im, cbar
        return im

    def venn2Diagram(self, set1: set, set2: set, labels=None):
        if labels is None:
            labels = ['Set1', 'Set2']
        print('set1:', set1)
        print('set2', set2)
        venn = venn2([set1, set2], set_labels=labels)
        venn.get_label_by_id('10').set_text('\n'.join(set1 - set2))
        if venn.get_label_by_id('11') is not None:
            venn.get_label_by_id('11').set_text('\n'.join(set1 & set2))
        venn.get_label_by_id('01').set_text('\n'.join(set2 - set1))
        c = venn2_circles([set1, set2], linestyle='solid')
        return venn

    def venn2DiagramWordCloud(self, set1: set, set2: set, labels=None):
        if labels is None:
            labels = ['Set1', 'Set2']
        print('set1:', set1)
        print('set2', set2)
        venn = venn2_wordcloud([set1, set2], set_labels=labels)
        venn.get_label_by_id('10').set_text('\n'.join(set1 - set2))
        if venn.get_label_by_id('11') is not None:
            venn.get_label_by_id('11').set_text('\n'.join(set1 & set2))
        venn.get_label_by_id('01').set_text('\n'.join(set2 - set1))
        # c = venn2_circles([set1, set2], linestyle='solid')
        return venn

    def venn3Diagram(self, set1: set, set2: set, set3: set, labels=None):
        if labels is None:
            labels = ['Set1', 'Set2', 'Set3']
        venn = venn3([set1, set2, set3], set_labels=labels)
        venn.get_label_by_id('100').set_text('\n'.join(set1 - set2 - set3))
        venn.get_label_by_id('110').set_text('\n'.join(set1 & set2 - set3))
        venn.get_label_by_id('010').set_text('\n'.join(set2 - set3 - set1))
        venn.get_label_by_id('101').set_text('\n'.join(set1 & set3 - set2))
        venn.get_label_by_id('111').set_text('\n'.join(set1 & set2 & set3))
        venn.get_label_by_id('011').set_text('\n'.join(set2 & set3 - set1))
        venn.get_label_by_id('001').set_text('\n'.join(set3 - set2 - set1))
        return venn
