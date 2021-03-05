import glob
import pandas as pd
import numpy as np


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
    out_file = 'results_feature_size.csv'
    accuracy_sum = {}
    c = 7
    for filename in files:
        size = int(filename.split('.')[0].split('_')[-1])
        df = pd.read_csv(filename)
        accuracy_sum[size] = 0
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
            accuracy = row[1]['ACC avg']
            if classifier_selected == classifier:
                # print(accuracy)
                accuracy_sum[size] += accuracy
    for item in accuracy_sum:
        accuracy_sum[item] = np.round(accuracy_sum[item] / c, 2)
    # print(accuracy_sum)
    df = pd.DataFrame.from_dict(accuracy_sum, orient="index")
    df.columns = ['Accuracy']
    df.to_csv(out_file)


FEATURE_SIZEs = [5, 10, 20, 30, 40, 50, 75, 100, 200]
# for s in FEATURE_SIZEs:
#     merge(s)
final_merge('Custom Voting')
