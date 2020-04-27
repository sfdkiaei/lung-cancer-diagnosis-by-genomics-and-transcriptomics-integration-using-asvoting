from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, log_loss, roc_curve
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt


# from SDAE.sdae import StackedDenoisingAE


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
        self.arr_train_pca = None
        self.arr_test_pca = None
        self.arr_train_kernel_pca = None
        self.arr_test_kernel_pca = None
        self.fv_sdae_train = None
        self.fv_sdae_val = None
        self.fv_sdae_test = None

    def getFeatureVectors(self):
        fv = {
            'fv_maf_frequent': self.fv_maf_frequent,
            'fv_maf_impact_high': self.fv_maf_impact_high,
            'fv_maf_impact_moderate': self.fv_maf_impact_moderate,
            'fv_maf_impact_low': self.fv_maf_impact_low,
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
            'pca': {
                'train': self.arr_train_pca,
                'test': self.arr_test_pca},
            'kernel_pca': {
                'train': self.arr_train_kernel_pca,
                'test': self.arr_test_kernel_pca},
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

    def PCA(self, X, y, X_test, n_components=2):
        """

        :param X:
        :param y:
        :param n_components: Number of components. If None, all non-zero components are kept.
        :return: X_new array-like, shape (n_samples, n_components)
        """
        pca = PCA(n_components=n_components)
        self.arr_train_pca = pca.fit_transform(X, y)
        self.arr_test_pca = pca.transform(X_test)
        print('[PCA] arr_test_pca created with shape:', self.arr_test_pca.shape)

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


class Analysis:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.gpc_accuracy = None
        self.gpc_auc = None
        self.gpc_log_loss = None
        self.gpc_predicted = None
        self.gpc_predicted_proba = None
        self.rfc_accuracy = None
        self.rfc_auc = None
        self.rfc_log_loss = None
        self.rfc_predicted = None
        self.rfc_predicted_proba = None
        self.mlp_accuracy = None
        self.mlp_auc = None
        self.mlp_log_loss = None
        self.mlp_predicted = None
        self.mlp_predicted_proba = None
        self.cnb_accuracy = None
        self.cnb_auc = None
        self.cnb_log_loss = None
        self.cnb_predicted = None
        self.cnb_predicted_proba = None
        self.gbc_accuracy = None
        self.gbc_auc = None
        self.gbc_log_loss = None
        self.gbc_predicted = None
        self.gbc_predicted_proba = None

    def plot_roc_curve(self, fpr, tpr, auc):
        """
        Example:
            probs = predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
            self.gpc_auc = roc_auc_score(y_test, probs)
            fpr, tpr, thresholds = roc_curve(y_test, probs)
            self.plot_roc_curve(fpr, tpr, auc)
        :param fpr:
        :param tpr:
        :param auc:
        :return:
        """
        plt.plot(fpr, tpr, color='orange', label='ROC (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def getAccuracies(self):
        acc = {
            'GaussianProcessClassifier': {
                'acc': self.gpc_accuracy,
                'auc': self.gpc_auc,
                'log_loss': self.gpc_log_loss},
            'RandomForestClassifier': {
                'acc': self.rfc_accuracy,
                'auc': self.rfc_auc,
                'log_loss': self.rfc_log_loss},
            'MLPClassifier': {
                'acc': self.mlp_accuracy,
                'auc': self.mlp_auc,
                'log_loss': self.mlp_log_loss},
            'ComplementNB': {
                'acc': self.cnb_accuracy,
                'auc': self.cnb_auc,
                'log_loss': self.cnb_log_loss},
            'GradientBoostingClassifier': {
                'acc': self.gbc_accuracy,
                'auc': self.gbc_auc,
                'log_loss': self.gbc_log_loss},
        }
        return acc

    def GaussianProcessClassifier(self, X_train, X_test, y_train, y_test):
        kernel = 1.0 * RBF(1.0)
        clf = GaussianProcessClassifier(kernel=kernel, random_state=110)
        clf.fit(X_train, y_train)

        self.gpc_predicted = clf.predict(X_test)
        self.gpc_predicted_proba = clf.predict_proba(X_test)
        self.gpc_accuracy = accuracy_score(y_test, self.gpc_predicted)
        self.gpc_log_loss = log_loss(y_test, self.gpc_predicted_proba)
        probs = self.gpc_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.gpc_auc = roc_auc_score(y_test, probs)
        # print("Test Score", gpc.score(X_test, y_test))  # equal to accuracy
        if self.verbose:
            print('-GaussianProcessClassifier:')
            print("Test Accuracy: %.4f"
                  % self.gpc_accuracy)
            print("Test AUC: %.4f"
                  % self.gpc_auc)
            print("Log Marginal Likelihood: %.4f"
                  % clf.log_marginal_likelihood(clf.kernel_.theta))
            print("Log-loss: %.4f"
                  % self.gpc_log_loss)

    def RandomForestClassifier(self, X_train, X_test, y_train, y_test):
        clf = RandomForestClassifier(max_depth=2, random_state=110)
        clf.fit(X_train, y_train)
        self.rfc_predicted = clf.predict(X_test)
        self.rfc_predicted_proba = clf.predict_proba(X_test)
        self.rfc_accuracy = accuracy_score(y_test, self.rfc_predicted)
        self.rfc_log_loss = log_loss(y_test, self.rfc_predicted_proba)
        probs = self.rfc_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.rfc_auc = roc_auc_score(y_test, probs)
        # fpr, tpr, thresholds = roc_curve(y_test, probs)
        # self.plot_roc_curve(fpr, tpr, self.rfc_auc)
        if self.verbose:
            print('-RandomForestClassifier:')
            print("Test Accuracy: %.4f"
                  % self.rfc_accuracy)
            print("Test AUC: %.4f"
                  % self.rfc_auc)
            print("Log-loss: %.4f"
                  % self.rfc_log_loss)

    def MLPClassifier(self, X_train, X_test, y_train, y_test):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 2), random_state=110)
        clf.fit(X_train, y_train)
        self.mlp_predicted = clf.predict(X_test)
        self.mlp_predicted_proba = clf.predict_proba(X_test)
        self.mlp_accuracy = accuracy_score(y_test, self.mlp_predicted)
        self.mlp_log_loss = log_loss(y_test, self.mlp_predicted_proba)
        probs = self.mlp_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.mlp_auc = roc_auc_score(y_test, probs)
        if self.verbose:
            print('-MLPClassifier:')
            print("Test Accuracy: %.4f"
                  % self.mlp_accuracy)
            print("Test AUC: %.4f"
                  % self.mlp_auc)
            print("Log-loss: %.4f"
                  % self.mlp_log_loss)

    def ComplementNB(self, X_train, X_test, y_train, y_test):
        clf = ComplementNB()
        clf.fit(X_train, y_train)
        self.cnb_predicted = clf.predict(X_test)
        self.cnb_predicted_proba = clf.predict_proba(X_test)
        self.cnb_accuracy = accuracy_score(y_test, self.cnb_predicted)
        self.cnb_log_loss = log_loss(y_test, self.cnb_predicted_proba)
        probs = self.cnb_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.cnb_auc = roc_auc_score(y_test, probs)
        if self.verbose:
            print('-ComplementNB:')
            print("Test Accuracy: %.4f"
                  % self.cnb_accuracy)
            print("Test AUC: %.4f"
                  % self.cnb_auc)
            print("Log-loss: %.4f"
                  % self.cnb_log_loss)

    def GradientBoostingClassifier(self, X_train, X_test, y_train, y_test):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=110)
        clf.fit(X_train, y_train)
        self.gbc_predicted = clf.predict(X_test)
        self.gbc_predicted_proba = clf.predict_proba(X_test)
        self.gbc_accuracy = accuracy_score(y_test, self.gbc_predicted)
        self.gbc_log_loss = log_loss(y_test, self.gbc_predicted_proba)
        probs = self.gbc_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.gbc_auc = roc_auc_score(y_test, probs)
        if self.verbose:
            print('-GradientBoostingClassifier:')
            print("Test Accuracy: %.4f"
                  % self.gbc_accuracy)
            print("Test AUC: %.4f"
                  % self.gbc_auc)
            print("Log-loss: %.4f"
                  % self.gbc_log_loss)
