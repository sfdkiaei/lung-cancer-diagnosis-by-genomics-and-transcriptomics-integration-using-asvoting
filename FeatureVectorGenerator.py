from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import numpy as np
import mygene
# from SDAE.sdae import StackedDenoisingAE
from Methods import StatisticalTest


class FeatureVectorGenerator:
    # fv: Feature Vector
    # arr: Array
    def __init__(self):
        self.fv_maf_frequent = None
        self.arr_train_maf_frequent = None
        self.arr_test_maf_frequent = None
        self.fv_maf_impact_high = None
        self.arr_train_maf_impact_high = None
        self.arr_test_maf_impact_high = None
        self.fv_maf_impact_moderate = None
        self.arr_train_maf_impact_moderate = None
        self.arr_test_maf_impact_moderate = None
        self.fv_maf_impact_low = None
        self.arr_train_maf_impact_low = None
        self.arr_test_maf_impact_low = None
        self.fv_maf_integrated = None
        self.arr_train_maf_integrated = None
        self.arr_test_maf_integrated = None
        self.arr_train_pca = None
        self.arr_test_pca = None
        self.arr_train_kernel_pca = None
        self.arr_test_kernel_pca = None
        self.fv_biomarker = None
        self.arr_train_biomarker = None
        self.arr_test_biomarker = None
        self.fv_sdae_train = None
        self.fv_sdae_val = None
        self.fv_sdae_test = None

    def getFeatureVectors(self):
        fv = {
            'fv_maf_frequent': self.fv_maf_frequent,
            'fv_maf_impact_high': self.fv_maf_impact_high,
            'fv_maf_impact_moderate': self.fv_maf_impact_moderate,
            'fv_maf_impact_low': self.fv_maf_impact_low,
            'fv_maf_integrated': self.fv_maf_integrated,
            'fv_biomarker': self.fv_biomarker,
            'fv_sdae_train': self.fv_sdae_train,
            'fv_sdae_val': self.fv_sdae_val,
            'fv_sdae_test': self.fv_sdae_test
        }
        return fv

    def getArrays(self):
        arr = {
            'maf_frequent': {
                'train': self.arr_train_maf_frequent,
                'test': self.arr_test_maf_frequent},
            'maf_impact_high': {
                'train': self.arr_train_maf_impact_high,
                'test': self.arr_test_maf_impact_high},
            'maf_impact_moderate': {
                'train': self.arr_train_maf_impact_moderate,
                'test': self.arr_test_maf_impact_moderate},
            'maf_impact_low': {
                'train': self.arr_train_maf_impact_low,
                'test': self.arr_test_maf_impact_low},
            'maf_integrated': {
                'train': self.arr_train_maf_integrated,
                'test': self.arr_test_maf_integrated},
            'pca': {
                'train': self.arr_train_pca,
                'test': self.arr_test_pca},
            'kernel_pca': {
                'train': self.arr_train_kernel_pca,
                'test': self.arr_test_kernel_pca},
            'biomarker': {
                'train': self.arr_train_biomarker,
                'test': self.arr_test_biomarker},
        }
        return arr

    @staticmethod
    def getFrequentValues(df, column_name, n):
        """
        Get top n most frequent values in column_name
        :param df:
        :param column_name:
        :param n:
        :return:
        """
        return df[column_name].value_counts()[:n].index.tolist()

    def MAFGenes(self, X_train, y_train, X_test, df_maf, size):
        """
        last column is label ("isTumor")
        :param df: Transcriptome Profiling DataFrame
        :param df_maf: MAF file DataFrame
        :param size: expected number of genes
        :return:
        """
        genes = self.getFrequentValues(df_maf, 'Gene', size)
        self.fv_maf_frequent = genes
        self.arr_train_maf_frequent = X_train[genes].values
        self.arr_test_maf_frequent = X_test[genes].values
        genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'HIGH'], 'Gene', size)
        self.fv_maf_impact_high = genes
        self.arr_train_maf_impact_high = X_train[genes].values
        self.arr_test_maf_impact_high = X_test[genes].values
        genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'MODERATE'], 'Gene', size)
        self.fv_maf_impact_moderate = genes
        self.arr_train_maf_impact_moderate = X_train[genes].values
        self.arr_test_maf_impact_moderate = X_test[genes].values
        genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'LOW'], 'Gene', size)
        self.fv_maf_impact_low = genes
        self.arr_train_maf_impact_low = X_train[genes].values
        self.arr_test_maf_impact_low = X_test[genes].values
        print('[MAFGenes] fv_maf_frequent created with size:', len(self.fv_maf_frequent))
        print('[MAFGenes] fv_maf_impact_high created with size:', len(self.fv_maf_impact_high))
        print('[MAFGenes] fv_maf_impact_moderate created with size:', len(self.fv_maf_impact_moderate))
        print('[MAFGenes] fv_maf_impact_low created with size:', len(self.fv_maf_impact_low))
        # genes = self.getFrequentValues(df_maf, 'Gene', size)
        # self.fv_maf_frequent = genes
        # genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'HIGH'], 'Gene', size)
        # self.fv_maf_impact_high = genes
        # genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'MODERATE'], 'Gene', size)
        # self.fv_maf_impact_moderate = genes
        # genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'LOW'], 'Gene', size)
        # self.fv_maf_impact_low = genes
        # print('[MAFGenes] fv_maf_frequent created with size:', len(self.fv_maf_frequent))
        # print('[MAFGenes] fv_maf_impact_high created with size:', len(self.fv_maf_impact_high))
        # print('[MAFGenes] fv_maf_impact_moderate created with size:', len(self.fv_maf_impact_moderate))
        # print('[MAFGenes] fv_maf_impact_low created with size:', len(self.fv_maf_impact_low))

        # genes = self.getFrequentValues(df_maf, 'Gene', size)
        # self.fv_maf_frequent = df[genes + ['isTumor']]
        # genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'HIGH'], 'Gene', size)
        # self.fv_maf_impact_high = df[genes + ['isTumor']]
        # genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'MODERATE'], 'Gene', size)
        # self.fv_maf_impact_moderate = df[genes + ['isTumor']]
        # genes = self.getFrequentValues(df_maf[df_maf['IMPACT'] == 'LOW'], 'Gene', size)
        # self.fv_maf_impact_low = df[genes + ['isTumor']]
        # print('[MAFGenes] fv_maf_frequent created with shape:', self.fv_maf_frequent.shape)
        # print('[MAFGenes] fv_maf_impact_high created with shape:', self.fv_maf_impact_high.shape)
        # print('[MAFGenes] fv_maf_impact_moderate created with shape:', self.fv_maf_impact_moderate.shape)
        # print('[MAFGenes] fv_maf_impact_low created with shape:', self.fv_maf_impact_low.shape)
        # return [
        #     self.fv_maf_frequent.shape,
        #     self.fv_maf_impact_high.shape,
        #     self.fv_maf_impact_moderate.shape,
        #     self.fv_maf_impact_low.shape,
        # ]

    def PCA(self, X, y, X_test=None, n_components=2):
        """

        :param X:
        :param y:
        :param n_components: Number of components. If None, all non-zero components are kept.
        :return: X_new array-like, shape (n_samples, n_components)
        """
        pca = PCA(n_components=n_components)
        self.arr_train_pca = pca.fit_transform(X, y)
        if X_test is not None:
            self.arr_test_pca = pca.transform(X_test)
            print('[PCA] arr_test_pca created with shape:', self.arr_test_pca.shape)
        return self.arr_train_pca, self.arr_test_pca

    def KernelPCA(self, X, y, X_test, n_components=2, kernel="rbf"):
        """

        :param X:
        :param y:
        :param n_components: Number of components. If None, all non-zero components are kept.
        :param kernel: “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
        :return: X_new array-like, shape (n_samples, n_components)
        """
        transformer = KernelPCA(n_components=n_components, kernel=kernel)
        self.arr_train_kernel_pca = transformer.fit_transform(X, y)
        self.arr_test_kernel_pca = transformer.transform(X_test)
        print('[KernelPCA] arr_test_kernel_pca created with shape:', self.arr_test_kernel_pca.shape)

    def integrateMAFGenes(self, X_train, y_train, X_test, sample_num: int):
        feature_vectors = {
            'fv_maf_frequent': self.fv_maf_frequent,
            'fv_maf_impact_high': self.fv_maf_impact_high,
            'fv_maf_impact_moderate': self.fv_maf_impact_moderate,
            'fv_maf_impact_low': self.fv_maf_impact_low
        }
        housekeepings = open('Data/genes_housekeeping.txt', 'r').read().split('\n')
        fv_housekeeping = []
        fv_housekeeping_symbol = []
        for gene in housekeepings:
            fv_housekeeping.append(gene.split(',')[1])  # Ensemble gene id
            fv_housekeeping_symbol.append(gene.split(',')[0])  # gene symbol
        st = StatisticalTest()
        samples_tumor = X_train[y_train == True].sample(n=sample_num)
        samples_normal = X_train[y_train == False].sample(n=sample_num)
        pvalues = {}
        for fv_name, fv_genes in feature_vectors.items():
            fv_len = len(fv_genes)
            pvalues[fv_name] = np.zeros((fv_len, len(fv_housekeeping)))
            for i in range(fv_len):
                for j in range(len(fv_housekeeping)):
                    pvalue = st.tTest(
                        samples_tumor.loc[:, fv_genes[i]].values.flatten().tolist(),
                        samples_normal.loc[:, fv_housekeeping[j]].values.flatten().tolist()
                    )
                    pvalues[fv_name][i, j] = pvalue
        fv_avg = []  # average p-value for each gene on 16 housekeeping genes
        fv_avg_genes = []
        for fv_name, fv_pvalue in pvalues.items():
            rows = fv_pvalue.shape[0]
            cols = fv_pvalue.shape[1]
            for idx_driver in range(0, rows):
                # pvalue_avg = np.average(fv_pvalue[idx_driver, :])
                # pvalue_median = np.median(fv_pvalue[idx_driver, :])
                fv_pvalue_temp = np.sort(fv_pvalue[idx_driver, :])
                pvalue_avg = np.average(
                    fv_pvalue_temp[:-4])  # average of housekeeping genes (except four greatest ones) p-value
                fv_avg.append(pvalue_avg)
                fv_avg_genes.append(feature_vectors[fv_name][idx_driver])

        fv_avg = np.array(fv_avg)
        fv_avg_genes = np.array(fv_avg_genes)
        # print(fv_avg)
        # print('######################################')
        # Remove repeated ones and sort from smallest p-value ascending
        _, fv_avg_distinct_indices = np.unique(fv_avg, return_index=True)
        fv_avg = fv_avg[fv_avg_distinct_indices]
        fv_avg_genes = fv_avg_genes[fv_avg_distinct_indices]
        try:
            smallest = fv_avg.argsort()[:fv_len]
        except Exception as exp:
            print('fv_avg size:', len(fv_avg), 'desired size:', fv_len)
            print(type(exp), exp.args)
        # print(smallest)
        # print('######################################')
        # print(fv_avg_genes[smallest])
        # mg = mygene.MyGeneInfo()
        # fv_avg_genes_symbol = mg.querymany(fv_avg_genes[smallest], scopes='ensembl.gene', fields='symbol',
        #                                    as_dataframe=True).loc[:,
        #                       'symbol'].values.flatten().tolist()
        # print(fv_avg_genes_symbol)
        self.fv_maf_integrated = fv_avg_genes[smallest]
        self.arr_train_maf_integrated = X_train[self.fv_maf_integrated].values
        self.arr_test_maf_integrated = X_test[self.fv_maf_integrated].values
        print('[integrateMAFGenes] fv_maf_integrated created with size:', len(self.fv_maf_integrated))

    def lungCancerBiomarkers(self, X_train, X_test):
        """
        :return:
        """
        genes = []
        with open('Data/biomarkers_lung.txt', 'r', encoding='utf8') as f:
            for line in f:
                if not line.startswith('#'):
                    genes.append(line.split(',')[1].strip())  # get Entrez gene id
        print(genes)

        self.fv_biomarker = genes
        self.arr_train_biomarker = X_train[genes].values
        self.arr_test_biomarker = X_test[genes].values

        print('[lungCancerBiomarkers] fv_biomarker created with size:', len(self.fv_biomarker))

    # def SDAE(self, X_train, X_test=None, X_validation=None, y=None, n_layers=2, n_hid=[10], dropout=[0.1], n_epoch=2,
    #          get_enc_model=False, write_model=False, dir_out='../output/'):
    #     """
    #     train a stacked denoising autoencoder and get the trained model,
    #     dense representations of the final hidden layer, and reconstruction mse
    #     :param X_train:
    #     :param X_test:
    #     :param X_validation:
    #     :param y:
    #     :param n_layers:
    #     :param n_hid:
    #     :param dropout:
    #     :param n_epoch:
    #     :param get_enc_model:
    #     :param write_model:
    #     :param dir_out:
    #     :return:
    #     """
    #     cur_sdae = StackedDenoisingAE(n_layers=n_layers, n_hid=n_hid, dropout=dropout, nb_epoch=n_epoch)
    #     model, (self.fv_sdae_train, self.fv_sdae_val, self.fv_sdae_test), \
    #     recon_mse = cur_sdae.get_pretrained_sda(X_train,
    #                                             X_validation,
    #                                             X_test,
    #                                             get_enc_model=get_enc_model,
    #                                             write_model=write_model,
    #                                             dir_out=dir_out)
