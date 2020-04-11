import json
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from SDAE.sdae import StackedDenoisingAE

plt.style.use('ggplot')
np.random.seed(110)


class Analysis:
    def __init__(self):  # fv: Feature Vector
        self.fv_maf = None
        self.fv_pca = None
        self.fv_kernel_pca = None
        self.fv_sdae_train = None
        self.fv_sdae_val = None
        self.fv_sdae_test = None
        self.fv_gaussian_process = None
        self.fv_random_forest = None
        self.fv_mlp = None
        self.fv_complement_nb = None
        self.fv_gradient_boosting = None

    def MAFGenes(self):
        pass

    def PCA(self, X, y=None, n_components=2):
        """

        :param X:
        :param y:
        :param n_components: Number of components. If None, all non-zero components are kept.
        :return: X_new array-like, shape (n_samples, n_components)
        """
        pca = PCA(n_components=n_components)
        self.fv_pca = pca.fit_transform(X, y)

    def KernelPCA(self, X, y=None, n_components=2, kernel="rbf"):
        """

        :param X:
        :param y:
        :param n_components: Number of components. If None, all non-zero components are kept.
        :param kernel: “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
        :return: X_new array-like, shape (n_samples, n_components)
        """
        transformer = KernelPCA(n_components=n_components, kernel=kernel)
        self.fv_kernel_pca = transformer.fit_transform(X, y)

    def SDAE(self, X_train, X_test=None, X_validation=None, y=None, n_layers=2, n_hid=[10], dropout=[0.1], n_epoch=2,
             get_enc_model=False, write_model=False, dir_out='../output/'):
        """
        train a stacked denoising autoencoder and get the trained model,
        dense representations of the final hidden layer, and reconstruction mse
        :param X_train:
        :param X_test:
        :param X_validation:
        :param y:
        :param n_layers:
        :param n_hid:
        :param dropout:
        :param n_epoch:
        :param get_enc_model:
        :param write_model:
        :param dir_out:
        :return:
        """
        cur_sdae = StackedDenoisingAE(n_layers=n_layers, n_hid=n_hid, dropout=dropout, nb_epoch=n_epoch)
        model, (self.fv_sdae_train, self.fv_sdae_val, self.fv_sdae_test), \
        recon_mse = cur_sdae.get_pretrained_sda(X_train,
                                                X_validation,
                                                X_test,
                                                get_enc_model=get_enc_model,
                                                write_model=write_model,
                                                dir_out=dir_out)

    def GaussianProcessClassifier(self):
        pass

    def RandomForestClassifier(self):
        pass

    def MLPClassifier(self):
        pass

    def ComplementNB(self):
        pass

    def GradientBoostingClassifier(self):
        pass


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


data_path = 'Data/'
path = data_path + 'LUAD/TP - HTSeq - FPKM-UQ/'
normal = 'solid tissue normal/'
tumor = 'primary tumor/'
mapper_path = data_path + 'mapping case id - sample UUID.json'
mapper_path_save = data_path
maf_path = data_path + 'LUAD/SNV - maf - muse/6f5cde97-d259-414f-8122-6d0d66f49b74/' \
                       'TCGA.LUAD.muse.6f5cde97-d259-414f-8122-6d0d66f49b74.DR-10.0.somatic.maf'
maf_column_names = [
    'Hugo_Symbol',
    'Entrez_Gene_Id',
    'Tumor_Sample_UUID',
    'Gene',
    'IMPACT',
    'case_id'
]
maf_path_save = data_path + 'LUAD/'
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

# analyzer = Analysis()
# analyzer.PCA(X_train, y_train, 500)
# analyzer.KernelPCA(X_train, y_train, 500, "rbf")
# print(analyzer.fv_pca.shape)
# print(analyzer.fv_kernel_pca.shape)
# print('##################################')
# print(X_train.values[:2, :])
# print(np.array(X_train.values[:2, :]))
# print(np.array(X_train.values[:2, 1::]))
# print(np.array(X_train.values[0, :]))
# print('##################################')
# print(X_train.iloc[:, :2].values)
# print('##################################')
# print(X_train[:2])
# analyzer.SDAE(np.array(X_train.values[:4, :]), np.array(X_test.values[:2, :]), np.array(X_test.values[:2, :]))
# analyzer.SDAE(X_train.iloc[:, :4].values, X_test.iloc[:, :2].values, X_test.iloc[:, :2].values)
# analyzer.SDAE(X_train.iloc[:, :4], X_test.iloc[:, :2], X_test.iloc[:, :2])
# print(analyzer.fv_sdae_train.shape)
# print(analyzer.fv_sdae_train)

# kernel = 1.0 * RBF(1.0)
# gpc = GaussianProcessClassifier(kernel=kernel, random_state=110)
# gpc.fit(X_train, y_train)
# print(gpc.score(X_train, y_train))
#
# # print(gpc.predict_proba(X_test))
# # print(gpc.predict(X_test))
#
#
# print("Log Marginal Likelihood: %.3f"
#       % gpc.log_marginal_likelihood(gpc.kernel_.theta))
# print("Accuracy: %.3f"
#       % (accuracy_score(y_test, gpc.predict(X_test))))
# print("Log-loss: %.3f"
#       % (log_loss(y_test, gpc.predict_proba(X_test))))

# clf = ComplementNB()
# clf.fit(X_train, y_train)
# print(clf.score(X_train, y_train))
# print("Accuracy: %.3f"
#       % (accuracy_score(y_test, clf.predict(X_test))))
# print("Log-loss: %.3f"
#       % (log_loss(y_test, clf.predict_proba(X_test))))

# clf = RandomForestClassifier(max_depth=2, random_state=110)
# clf.fit(X_train, y_train)
# print(clf.score(X_train, y_train))
# print("Accuracy: %.3f"
#       % (accuracy_score(y_test, clf.predict(X_test))))
# print("Log-loss: %.3f"
#       % (log_loss(y_test, clf.predict_proba(X_test))))

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(500, 2), random_state=110)
# clf.fit(X_train, y_train)
# print(clf.score(X_train, y_train))
# print("Accuracy: %.3f"
#       % (accuracy_score(y_test, clf.predict(X_test))))
# print("Log-loss: %.3f"
#       % (log_loss(y_test, clf.predict_proba(X_test))))

# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
#                                  max_depth=1, random_state=110)
# clf.fit(X_train, y_train)
# print(clf.score(X_train, y_train))
# print("Accuracy: %.3f"
#       % (accuracy_score(y_test, clf.predict(X_test))))
# print("Log-loss: %.3f"
#       % (log_loss(y_test, clf.predict_proba(X_test))))
