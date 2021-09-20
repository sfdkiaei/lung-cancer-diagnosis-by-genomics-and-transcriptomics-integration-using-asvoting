import glob
import pandas as pd
import numpy as np
from datetime import datetime


def merge(feature_size):
    FEATURE_SIZE = feature_size
    folders = glob.glob('result_' + str(FEATURE_SIZE) + '_*')
    out_file = 'final_results_' + str(FEATURE_SIZE) + '.csv'
    size = 0

    dfs = []
    for folder in folders:
        df = pd.read_excel(folder + '/results.xlsx')
        dfs.append(df)
        size += 1
    df = pd.concat(dfs, ignore_index=True)
    # print(df.groupby(['Acc']).dir.mean().reset_index())

    # print(df.loc[0, 'Acc'])
    acc = {}
    tpr = {}
    tnr = {}
    ppv = {}
    for row in df.iterrows():
        if (row[1]['Feature Vector'], row[1]['Classifier']) not in acc:
            acc[(row[1]['Feature Vector'], row[1]['Classifier'])] = []
            tpr[(row[1]['Feature Vector'], row[1]['Classifier'])] = []
            tnr[(row[1]['Feature Vector'], row[1]['Classifier'])] = []
            ppv[(row[1]['Feature Vector'], row[1]['Classifier'])] = []
        acc[(row[1]['Feature Vector'], row[1]['Classifier'])].append(row[1]['Acc'])
        tpr[(row[1]['Feature Vector'], row[1]['Classifier'])].append(row[1]['TPR'])
        tnr[(row[1]['Feature Vector'], row[1]['Classifier'])].append(row[1]['TNR'])
        ppv[(row[1]['Feature Vector'], row[1]['Classifier'])].append(row[1]['PPV'])

    for item in acc:
        acc[item] = (np.round(np.array(acc[item]).mean(), 2), np.round(np.array(acc[item]).std(), 2))
        tpr[item] = (np.round(np.array(tpr[item]).mean(), 2), np.round(np.array(tpr[item]).std(), 2))
        tnr[item] = (np.round(np.array(tnr[item]).mean(), 2), np.round(np.array(tnr[item]).std(), 2))
        ppv[item] = (np.round(np.array(ppv[item]).mean(), 2), np.round(np.array(ppv[item]).std(), 2))
        # print(item, acc[item])
    # print('Total Execution time:', size)
    # print(acc)
    # print(tpr)
    # print(tnr)
    # print(ppv)
    df_acc = pd.DataFrame.from_dict(acc, orient="index")
    df_tpr = pd.DataFrame.from_dict(tpr, orient="index")
    df_tnr = pd.DataFrame.from_dict(tnr, orient="index")
    df_ppv = pd.DataFrame.from_dict(ppv, orient="index")
    df = pd.concat([df_acc, df_tpr, df_tnr, df_ppv], axis=1)
    df.columns = ['ACC avg', 'ACC std', 'TPR avg', 'TPR std', 'TNR avg', 'TNR std', 'PPV avg', 'PPV std']
    df.to_csv(out_file)
    print(out_file, 'created from', size, 'files.')


def final_merge(classifier_selected):
    files = glob.glob('final_results_*.csv')
    out_file = f'results_{classifier_selected}_feature_size_{datetime.now().date()}.csv'
    accuracy_sum = {}
    tpr_sum = {}
    tnr_sum = {}
    ppv_sum = {}
    accuracy_std_sum = {}
    tpr_std_sum = {}
    tnr_std_sum = {}
    ppv_std_sum = {}
    c = 7.0
    for filename in files:
        size = int(filename.split('.')[0].split('_')[-1])
        df = pd.read_csv(filename)
        accuracy_sum[size] = 0
        tpr_sum[size] = 0
        tnr_sum[size] = 0
        ppv_sum[size] = 0
        accuracy_std_sum[size] = 0
        tpr_std_sum[size] = 0
        tnr_std_sum[size] = 0
        ppv_std_sum[size] = 0
        # data[size] = df
        for row in df.iterrows():
            name = row[1][0].replace('\'', '')
            name = name.replace('(', '')
            name = name.replace(')', '')
            feature = name.split(',')[0]
            classifier = name.split(',')[1][1:]
            # accuracy = str(row[1]['ACC avg']) + ' ± ' + str(row[1]['ACC std'])
            # tpr = str(row[1]['TPR avg']) + ' ± ' + str(row[1]['TPR std'])
            # tnr = str(row[1]['TNR avg']) + ' ± ' + str(row[1]['TNR std'])
            # ppv = str(row[1]['PPV avg']) + ' ± ' + str(row[1]['PPV std'])
            tpr = row[1]['TPR avg']
            tnr = row[1]['TNR avg']
            ppv = row[1]['PPV avg']
            accuracy = row[1]['ACC avg']
            tpr_std = row[1]['TPR std']
            tnr_std = row[1]['TNR std']
            ppv_std = row[1]['PPV std']
            accuracy_std = row[1]['ACC std']
            if classifier_selected == classifier and feature != 'biomarker':
                accuracy_sum[size] += accuracy
                tpr_sum[size] += tpr
                tnr_sum[size] += tnr
                ppv_sum[size] += ppv
                accuracy_std_sum[size] += accuracy_std
                tpr_std_sum[size] += tpr_std
                tnr_std_sum[size] += tnr_std
                ppv_std_sum[size] += ppv_std
                # print(size, accuracy, accuracy_sum[size])
    for item in accuracy_sum:
        accuracy_sum[item] = np.round(accuracy_sum[item] / c, 2)
        tpr_sum[item] = np.round(tpr_sum[item] / c, 2)
        tnr_sum[item] = np.round(tnr_sum[item] / c, 2)
        ppv_sum[item] = np.round(ppv_sum[item] / c, 2)
        accuracy_std_sum[item] = np.round(accuracy_std_sum[item] / c, 2)
        tpr_std_sum[item] = np.round(tpr_std_sum[item] / c, 2)
        tnr_std_sum[item] = np.round(tnr_std_sum[item] / c, 2)
        ppv_std_sum[item] = np.round(ppv_std_sum[item] / c, 2)
    # print(accuracy_sum)
    # df = pd.DataFrame.from_dict([accuracy_sum, tpr_sum, tnr_sum], orient="index")
    df = pd.DataFrame([accuracy_sum, tpr_sum, tnr_sum, ppv_sum, accuracy_std_sum, tpr_std_sum, tnr_std_sum, ppv_std_sum]).T
    df = df.sort_index()
    df.columns = ['Accuracy', 'TPR', 'TNR', 'PPV', 'Accuracy std', 'TPR std', 'TNR std', 'PPV std']
    df.to_csv(out_file)
    print(out_file, 'generated successfully')


def get_most_consistent_genes():
    out_file = f'selected_genes_info_{datetime.now().date()}.csv'
    files = []
    genes = {}
    # files = glob.glob('result_250*/fv_maf_integrated.txt')
    for fs in FEATURE_SIZEs:
        files += glob.glob('result_' + str(fs) + '_*/fv_maf_*.txt')
        files += glob.glob('result_' + str(fs) + '_*/fv_maf_integrated.txt')
    print(files)
    for file in files:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                gene = line.split(',')[1].strip()
                if gene not in genes:
                    genes[gene] = 0
                genes[gene] += 1
    print(genes)
    df = pd.DataFrame.from_dict(genes, orient="index")
    df.columns = ['count']
    df.to_csv(out_file)
    genes = sorted(genes.items(), key=lambda x: x[1], reverse=True)
    print(genes)


FEATURE_SIZEs = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500]
# for s in FEATURE_SIZEs:
#     merge(s)
# final_merge('Custom Voting')

classifiers = ['Gaussian Process', 'Random Forest', 'MLP', 'Gradient Boosting', 'Non Linear SVM', 'Fuzzy Pattern',
               'Fuzzy Pattern GA', 'MaxVoting', 'Custom Voting']
for item in classifiers:
    final_merge(item)

# get_most_consistent_genes()
