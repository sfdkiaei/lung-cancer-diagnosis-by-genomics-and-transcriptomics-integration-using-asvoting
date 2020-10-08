import time

from sklearn import preprocessing

from Analysis import Analysis
from BalancingDataset import BalancingDataset
from FeatureVectorGenerator import FeatureVectorGenerator
from Methods import StatisticalTest
from Methods import Visualization
from Methods import Fusion
import json
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
from functools import reduce
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import mygene
from sklearn.model_selection import ShuffleSplit

warnings.filterwarnings("ignore")
plt.style.use('ggplot')
np.random.seed(110)


def createSampleDataFrame(files_exp, save=False, save_name="fullData", is_tumor=None):
    """
    Merge all transcriptome profiling data in one dataFrame and save them as .pkl file
    :param files_exp: String
    :param save: Boolean
    :param save_name: path + name of saving file
    :param is_tumor: True/False
    :return: Full dataFrame. Columns: genes, Row: expression values, Index: Sample UUID
    """
    all_files = glob.glob(files_exp)
    print("Total number of samples:", len(all_files))

    data_frames = []
    hold = 0
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, delimiter='\t')
        # filename example: Data/LUAD/TP - HTSeq - FPKM-UQ/
        # solid tissue normal\3c6e35a0-0dd3-4d28-b191-91f5ce1cca08\d179f9d5-0b8c-459e-9f47-271b25887c3c.FPKM-UQ.txt.gz
        uuid = filename.split('\\')[-1].split('.')[-4]
        df.columns = ['UUID', uuid]
        if df.shape != hold:
            hold = df.shape
            print(df.shape)
        data_frames.append(df)

    df_full = reduce(lambda left, right: pd.merge(left, right, on=['UUID'], how='outer'), data_frames)
    df_full = df_full.set_index('UUID').T
    df_full.rename(columns=lambda x: x.split('.')[0], inplace=True)  # remove numbers after '.' for genes
    if is_tumor is not None:
        df_full['isTumor'] = [is_tumor] * df_full.shape[0]  # primary tumor / solid tissue normal

    print("Final data frame size:", df_full.shape)
    print("This has", df_full.isna().sum().sum(), "NaN values.")
    print(df_full.head())
    if save:
        df_full.to_pickle(save_name + ".pkl")
    return df_full


def createSampleCaseMapper(mapper_file_path, save=False, save_name="sampleCaseMapper"):
    """
    Create DataFrame of sample UUID and corresponding case id and save them as .pkl file.
    :param mapper_file_path: String. Download it from TCGA repository JSON button after filtering your data
    :param save: Boolean
    :param save_name: path + name of saving file
    :return: Sample UUID and Case Id mapper dataFrame. Columns: [UUID, CaseId]
    """
    mapper_file = open(mapper_file_path)
    mapper = json.load(mapper_file)
    df = pd.DataFrame(columns=["UUID", "CaseId"])
    unusual_sample_ids = []
    s = 0
    for idx, item in enumerate(mapper):
        if len(item['cases']) > 1:
            s += 1
            unusual_sample_ids.append(idx)
        uuid = item['file_name'].split('.')[0]
        case_id = item['cases'][0]['case_id']
        df.loc[idx] = [uuid, case_id]
    print(s, 'sample with more than one case exist.')

    print("Final data frame size:", df.shape)
    print("This has", df.isna().sum().sum(), "NaN values.")
    print(df.head())
    if save:
        df.to_pickle(save_name + ".pkl")
    return df, unusual_sample_ids


def loadMAF(file_path, maf_column_names=None, save=False, save_name="maf"):
    """
    Load MAF file into DataFrame and save it as .pkl file.
    selecting desired columns is optional
    :param file_path: String. maf file path
    :param columns: String of excel columns. example: 'A:C,E,H'
    :param save: Boolean
    :param save_name: path + name of saving file
    :return: Sample UUID and Case Id mapper dataFrame. Columns: [UUID, CaseId]
    """
    if maf_column_names is None:
        # columns = get all column names
        return None
    df = pd.read_csv(file_path, skip_blank_lines=True, low_memory=False, header=1, comment="#", sep="\t")
    for col in df.columns:
        if col not in maf_column_names:
            del df[col]

    print("Final data frame size:", df.shape)
    print("This has", df.isna().sum().sum(), "NaN values.")
    print(df.isna)
    print(df.head())
    if save:
        df.to_pickle(save_name + ".pkl")
    return df


def printRowsContainingNan(dataFrame):
    df1 = dataFrame[dataFrame.isna().any(axis=1)]
    print(df1.to_string())


def runStatisticsTest(X, y, feature_vectors, sample_num):
    center = 0.05
    housekeepings = open('Data/genes_housekeeping.txt', 'r').read().split('\n')
    fv_housekeeping = []
    fv_housekeeping_symbol = []
    for gene in housekeepings:
        fv_housekeeping.append(gene.split(',')[1])  # Ensemble gene id
        fv_housekeeping_symbol.append(gene.split(',')[0])  # gene symbol
    mg = mygene.MyGeneInfo()
    st = StatisticalTest()
    for feature_vector in tqdm(feature_vectors):
        fv = feature_vectors[feature_vector]
        if fv is not None:
            fv_len = len(fv)
            fv_symbol = mg.querymany(fv, scopes='ensembl.gene', fields='symbol', as_dataframe=True).loc[:,
                        'symbol'].values.flatten().tolist()
            pvalues = np.zeros((fv_len, len(fv_housekeeping)))
            samples_tumor = X[y == True].sample(n=sample_num)
            samples_normal = X[y == False].sample(n=sample_num)
            for i in range(fv_len):
                for j in range(len(fv_housekeeping)):
                    pvalue = st.tTest(
                        samples_tumor.loc[:, fv[i]].values.flatten().tolist(),
                        samples_normal.loc[:, fv_housekeeping[j]].values.flatten().tolist()
                    )
                    pvalues[i, j] = pvalue
            fig, ax = plt.subplots(figsize=(7, 12))
            visualization = Visualization()
            im = visualization.heatmap(pvalues, fv_symbol, fv_housekeeping_symbol, ax=ax,
                                       cmap="plasma", cbarlabel="P-Value", center=center)
            plt.title(feature_vector)
            fig.tight_layout()
            plt.savefig(feature_vector + '-pvalue_' + str(center) + '.png')
            plt.show()


def splitData(df_tp, df_test=None, normalize=True):
    X = df_tp.iloc[:, :-1]
    y = df_tp.iloc[:, -1]
    X, y = balancer.overSampling(X, y, 'ADASYN')
    if normalize:
        print('Normalizing data...')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    if df_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
    else:
        X_train = X
        y_train = y
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]
        X_test, y_test = balancer.underSampling(X_test, y_test, 'RandomUnderSampler')
        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler()
            X_test_scaled = min_max_scaler.fit_transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    print('Train data contains', len(y_train[y_train == True]), 'tumor', len(y_train[y_train == False]),
          'normal samples.')
    print('Test data contains', len(y_test[y_test == True]), 'tumor', len(y_test[y_test == False]),
          'normal samples.')
    cv = ShuffleSplit(n_splits=5, test_size=0.3)
    return X_train, X_test, y_train, y_test, cv


def analyzeData(df_tp, df_maf, df_test=None, normalize=True, save=False, statisticsTest=False,
                statisticsTestSampleNum=40, featureSize=50):
    X_train, X_test, y_train, y_test, cv = splitData(df_tp, df_test, normalize=normalize)

    generator = FeatureVectorGenerator()
    generator.MAFGenes(X_train, y_train, X_test, df_maf, size=featureSize)
    generator.PCA(X_train, y_train, X_test, featureSize)
    generator.KernelPCA(X_train, y_train, X_test, featureSize, "rbf")

    if statisticsTest:
        feature_vectors = generator.getFeatureVectors()
        runStatisticsTest(X_test, y_test, feature_vectors, statisticsTestSampleNum)

    result = []
    for key, value in tqdm(generator.getArrays().items()):
        if value['train'] is not None:
            # print('----------------', 'Feature Vector:', key)
            analyzer = Analysis(cv, verbose=False)
            # analyzer.ComplementNB(value['train'], value['test'], y_train, y_test)
            model_gpc = analyzer.GaussianProcessClassifier(value['train'], value['test'], y_train, y_test)
            model_rfc = analyzer.RandomForestClassifier(value['train'], value['test'], y_train, y_test)
            model_gbc = analyzer.GradientBoostingClassifier(value['train'], value['test'], y_train, y_test)
            model_mlp = analyzer.MLPClassifier(value['train'], value['test'], y_train, y_test)
            # model_tpot = analyzer.TpotClassifier(value['train'], value['test'], y_train, y_test)
            model_nlsvm = analyzer.NonLinearSVMClassifier(value['train'], value['test'], y_train, y_test)
            model_fpc = analyzer.FuzzyPatternClassifier(value['train'], value['test'], y_train, y_test)
            model_fpcga = analyzer.FuzzyPatternClassifierGA(value['train'], value['test'], y_train, y_test)
            analyzer.maxVoting(y_test)
            # analyzer.weightedMaxVoting(y_test, [1, 1, 1, 1, 1, 1, 1])
            analyzer.customVoting(y_test, size=5, threshold_tpr=0.6, threshold_tnr=0.6)

            if save:
                analyzer.saveModel(model_gpc, 'GaussianProcessClassifier')
                analyzer.saveModel(model_rfc, 'RandomForestClassifier')
                analyzer.saveModel(model_gbc, 'GradientBoostingClassifier')
                analyzer.saveModel(model_mlp, 'MLPClassifier')
                # analyzer.saveModel(model_tpot, 'TpotClassifier')
                analyzer.saveModel(model_nlsvm, 'NonLinearSVMClassifier')
                analyzer.saveModel(model_fpc, 'FuzzyPatternClassifier')
                analyzer.saveModel(model_fpcga, 'FuzzyPatternClassifierGA')
            for classifier, score in analyzer.getAccuracies().items():
                if score['acc'] is not None:
                    result.append([key, classifier,
                                   round(score['acc'], 4),
                                   round(score['auc'], 4),
                                   round(score['measurements']['TPR'], 4),
                                   round(score['measurements']['TNR'], 4),
                                   round(score['measurements']['PPV'], 4),
                                   round(score['log_loss'], 4)
                                   ])
                    # print(accuracy, classifier)
    result = pd.DataFrame(result, columns=['Feature Vector', 'Classifier',
                                           'Accuracy',
                                           'AUC',
                                           'Sensitivity(TPR)',
                                           'Specificity(TNR)',
                                           'Precision(PPV)',
                                           'Log Loss'
                                           ])
    print(result)
    if save:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        result.to_csv("result_" + timestr + ".csv")


if __name__ == "__main__":
    data_path = 'Data/'
    path_luad = data_path + 'LUAD/TP - HTSeq - FPKM-UQ/'
    path_lusc = data_path + 'LUSC/TP - HTSeq - FPKM-UQ/'
    normal = 'solid tissue normal/'
    tumor = 'primary tumor/'
    mapper_path = data_path + 'mapping case id - sample UUID.json'
    mapper_path_save = data_path
    maf_path = data_path + 'LUAD/SNV - maf - muse/6f5cde97-d259-414f-8122-6d0d66f49b74/' \
                           'TCGA.LUAD.muse.6f5cde97-d259-414f-8122-6d0d66f49b74.DR-10.0.somatic.maf'
    maf_path_save = data_path + 'LUAD/'
    maf_column_names = [
        'Hugo_Symbol',
        'Entrez_Gene_Id',
        'Tumor_Sample_UUID',
        'Gene',
        'IMPACT',
        'case_id'
    ]
    # createSampleDataFrame(path_luad + normal + '*/*.txt.gz', True, path_luad + normal + 'data', is_tumor=False)
    # createSampleDataFrame(path_luad + tumor + '*/*.txt.gz', True, path_luad + tumor + 'data', is_tumor=True)
    # createSampleCaseMapper(mapper_path, True, mapper_path_save + 'sampleCaseMapper')
    # loadMAF(maf_path, maf_column_names, True, maf_path_save + 'maf')
    df_normal = pd.read_pickle(path_luad + normal + 'data.pkl')
    df_tumor = pd.read_pickle(path_luad + tumor + 'data.pkl')
    df_tp = df_tumor.append(df_normal)  # Transcriptome Profiling
    mapper = pd.read_pickle(mapper_path_save + 'sampleCaseMapper.pkl')  # Sample UUID - Case Id mapper
    df_maf = pd.read_pickle(maf_path_save + 'maf.pkl')
    df_maf = df_maf[df_maf['Gene'].notnull()]  # drop rows which gene is none

    # createSampleDataFrame(path_lusc + normal + '*/*.txt.gz', True, path_lusc + normal + 'data', is_tumor=False)
    # createSampleDataFrame(path_lusc + tumor + '*/*.txt.gz', True, path_lusc + tumor + 'data', is_tumor=True)
    df_lusc_normal = pd.read_pickle(path_lusc + normal + 'data.pkl')
    df_lusc_tumor = pd.read_pickle(path_lusc + tumor + 'data.pkl')
    df_lusc_tp = df_lusc_tumor.append(df_lusc_normal)  # Transcriptome Profiling

    balancer = BalancingDataset()

    # analyzer = Analysis(verbose=False)
    analyzeData(df_tp, df_maf, df_test=df_lusc_tp, save=True, statisticsTest=False)

    # evaluateData(df_lusc_tp)

    # X = df_tp.iloc[:, :-1]
    # y = df_tp.iloc[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=110, test_size=0.3)
    # print(X_train.columns)
    # print(X_train)
    # print(y_train)

    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')

    # plt.figure(figsize=(16, 8))
    # plt.plot(df_tumor.iloc[:, 1::].mean(axis=1))
    # plt.plot(df_normal.iloc[:, 1::].mean(axis=1))
    # plt.title('Mean')
    # plt.xlabel('Genes')
    # plt.ylabel('Expression')
    # plt.legend(['Tumor', 'Normal'])
    # plt.tight_layout()
    # plt.savefig('Transcriptome - mean')
    # plt.show()

    # print(X_train.values[:2, :])
    # print(np.array(X_train.values[:2, :]))
    # print(np.array(X_train.values[:2, 1::]))
    # print(np.array(X_train.values[0, :]))
    # print('##################################')
    # print(X_train.iloc[:, :2].values)
    # print('##################################')
    # print(X_train[:2])
    # generator.SDAE(np.array(X_train.values[:4, :]), np.array(X_test.values[:2, :]), np.array(X_test.values[:2, :]))
    # generator.SDAE(X_train.iloc[:, :4].values, X_test.iloc[:, :2].values, X_test.iloc[:, :2].values)
    # generator.SDAE(X_train.iloc[:, :4], X_test.iloc[:, :2], X_test.iloc[:, :2])
    # print(generator.fv_sdae_train.shape)
    # print(generator.fv_sdae_train)
