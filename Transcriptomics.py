from sklearn import preprocessing
from Methods import FeatureVectorGenerator
from Methods import Analysis
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


def analyzeData(df_tp, df_maf, normalize=True):
    X = df_tp.iloc[:, :-1]
    y = df_tp.iloc[:, -1]
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=110, test_size=0.3)
    result = []

    generator = FeatureVectorGenerator()
    generator.MAFGenes(X_train, y_train, X_test, df_maf, size=200)
    generator.PCA(X_train, y_train, X_test, 100)
    generator.KernelPCA(X_train, y_train, X_test, 100, "rbf")
    for key, value in tqdm(generator.getArrays().items()):
        if value['train'] is not None:
            # print('----------------', 'Feature Vector:', key)
            analyzer = Analysis(verbose=False)
            # analyzer.ComplementNB(value['train'], value['test'], y_train, y_test)
            analyzer.GaussianProcessClassifier(value['train'], value['test'], y_train, y_test)
            analyzer.RandomForestClassifier(value['train'], value['test'], y_train, y_test)
            analyzer.GradientBoostingClassifier(value['train'], value['test'], y_train, y_test)
            analyzer.MLPClassifier(value['train'], value['test'], y_train, y_test)
            for classifier, score in analyzer.getAccuracies().items():
                if score['acc'] is not None:
                    result.append([key, classifier, (round(score['acc'], 4) * 100), round(score['log_loss'], 3)])
                    # print(accuracy, classifier)
    result = pd.DataFrame(result, columns=['Feature Vector', 'Classifier', 'Accuracy', 'Log Loss'])
    print(result)


data_path = 'Data/'
path = data_path + 'LUAD/TP - HTSeq - FPKM-UQ/'
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
# createSampleDataFrame(path + normal + '*/*.txt.gz', True, path + normal + 'data', is_tumor=False)
# createSampleDataFrame(path + tumor + '*/*.txt.gz', True, path + tumor + 'data', is_tumor=True)
# createSampleCaseMapper(mapper_path, True, mapper_path_save + 'sampleCaseMapper')
# loadMAF(maf_path, maf_column_names, True, maf_path_save + 'maf')
df_normal = pd.read_pickle(path + normal + 'data.pkl')
df_tumor = pd.read_pickle(path + tumor + 'data.pkl')
df_tp = df_tumor.append(df_normal)  # Transcriptome Profiling
mapper = pd.read_pickle(mapper_path_save + 'sampleCaseMapper.pkl')  # Sample UUID - Case Id mapper
df_maf = pd.read_pickle(maf_path_save + 'maf.pkl')
df_maf = df_maf[df_maf['Gene'].notnull()]  # drop rows which gene is none

analyzeData(df_tp, df_maf)

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
