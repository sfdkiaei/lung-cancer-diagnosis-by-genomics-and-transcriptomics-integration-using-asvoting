import matplotlib
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_curve
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_auc_score
from sklearn import svm
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC
from imblearn.over_sampling import SMOTE
from scipy import stats


# from SDAE.sdae import StackedDenoisingAE


class BalancingDataset:
    def __init__(self):
        pass

    def overSampling(self, X, y, method):
        """

        :param X:
        :param y:
        :param method: RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC
        :return:
        """
        if method == 'RandomOverSampler':
            X_resampled, y_resampled = RandomOverSampler().fit_resample(X, y)
        elif method == 'ADASYN':
            X_resampled, y_resampled = ADASYN().fit_resample(X, y)
        elif method == 'SMOTE':
            X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        elif method == 'BorderlineSMOTE':
            X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
        elif method == 'SVMSMOTE':
            X_resampled, y_resampled = SVMSMOTE().fit_resample(X, y)
        elif method == 'KMeansSMOTE':
            X_resampled, y_resampled = KMeansSMOTE().fit_resample(X, y)
        elif method == 'SMOTENC':
            smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)
            X_resampled, y_resampled = smote_nc.fit_resample(X, y)
        else:
            raise ValueError(method + ' not exists')
        return X_resampled, y_resampled

    def underSampling(self, X, y, method):
        """

        :param X:
        :param y:
        :param method: ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours,
        RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection,
        NeighbourhoodCleaningRule, InstanceHardnessThreshold
        :return:
        """
        if method == 'ClusterCentroids':
            X_resampled, y_resampled = ClusterCentroids().fit_resample(X, y)
        elif method == 'RandomUnderSampler':
            X_resampled, y_resampled = RandomUnderSampler().fit_resample(X, y)
        elif method == 'NearMiss':
            nm1 = NearMiss(version=1)
            X_resampled_nm1, y_resampled = nm1.fit_resample(X, y)
        elif method == 'EditedNearestNeighbours':
            X_resampled, y_resampled = EditedNearestNeighbours().fit_resample(X, y)
        elif method == 'RepeatedEditedNearestNeighbours':
            X_resampled, y_resampled = RepeatedEditedNearestNeighbours().fit_resample(X, y)
        elif method == 'AllKNN':
            X_resampled, y_resampled = AllKNN().fit_resample(X, y)
        elif method == 'CondensedNearestNeighbour':
            X_resampled, y_resampled = CondensedNearestNeighbour().fit_resample(X, y)
        elif method == 'OneSidedSelection':
            X_resampled, y_resampled = OneSidedSelection().fit_resample(X, y)
        elif method == 'NeighbourhoodCleaningRule':
            X_resampled, y_resampled = NeighbourhoodCleaningRule().fit_resample(X, y)
        elif method == 'InstanceHardnessThreshold':
            iht = InstanceHardnessThreshold(estimator=LogisticRegression(solver='lbfgs', multi_class='auto'))
            X_resampled, y_resampled = iht.fit_resample(X, y)
        else:
            raise ValueError(method + ' not exists')
        return X_resampled, y_resampled

    def overAndUnderSampling(self, X, y, method):
        """

        :param X:
        :param y:
        :param method: SMOTEENN, SMOTETomek
        :return:
        """
        if method == 'SMOTEENN':
            X_resampled, y_resampled = SMOTEENN().fit_resample(X, y)
        elif method == 'SMOTETomek':
            X_resampled, y_resampled = SMOTETomek().fit_resample(X, y)
        else:
            raise ValueError(method + ' not exists')
        return X_resampled, y_resampled


class StatisticalTest:
    def __init__(self):
        pass

    def tTest(self, a, b):
        statistic, pvalue = stats.ttest_ind(a, b, equal_var=False)
        return pvalue


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
        self.gpc = None
        self.gpc_measurements = None
        self.gpc_accuracy = None
        self.gpc_auc = None
        self.gpc_log_loss = None
        self.gpc_predicted = None
        self.gpc_predicted_proba = None
        self.rfc = None
        self.rfc_measurements = None
        self.rfc_accuracy = None
        self.rfc_auc = None
        self.rfc_log_loss = None
        self.rfc_predicted = None
        self.rfc_predicted_proba = None
        self.mlp = None
        self.mlp_measurements = None
        self.mlp_accuracy = None
        self.mlp_auc = None
        self.mlp_log_loss = None
        self.mlp_predicted = None
        self.mlp_predicted_proba = None
        self.cnb = None
        self.cnb_measurements = None
        self.cnb_accuracy = None
        self.cnb_auc = None
        self.cnb_log_loss = None
        self.cnb_predicted = None
        self.cnb_predicted_proba = None
        self.gbc = None
        self.gbc_measurements = None
        self.gbc_accuracy = None
        self.gbc_auc = None
        self.gbc_log_loss = None
        self.gbc_predicted = None
        self.gbc_predicted_proba = None
        self.nlsvm = None
        self.nlsvm_measurements = None
        self.nlsvm_accuracy = None
        self.nlsvm_auc = None
        self.nlsvm_log_loss = None
        self.nlsvm_predicted = None
        self.nlsvm_predicted_proba = None

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
                'measurements': self.gpc_measurements,
                'acc': self.gpc_accuracy,
                'auc': self.gpc_auc,
                'log_loss': self.gpc_log_loss},
            'RandomForestClassifier': {
                'measurements': self.rfc_measurements,
                'acc': self.rfc_accuracy,
                'auc': self.rfc_auc,
                'log_loss': self.rfc_log_loss},
            'MLPClassifier': {
                'measurements': self.mlp_measurements,
                'acc': self.mlp_accuracy,
                'auc': self.mlp_auc,
                'log_loss': self.mlp_log_loss},
            'ComplementNB': {
                'measurements': self.cnb_measurements,
                'acc': self.cnb_accuracy,
                'auc': self.cnb_auc,
                'log_loss': self.cnb_log_loss},
            'GradientBoostingClassifier': {
                'measurements': self.gbc_measurements,
                'acc': self.gbc_accuracy,
                'auc': self.gbc_auc,
                'log_loss': self.gbc_log_loss},
            'NonLinearSVMClassifier': {
                'measurements': self.nlsvm_measurements,
                'acc': self.nlsvm_accuracy,
                'auc': self.nlsvm_auc,
                'log_loss': self.nlsvm_log_loss},
        }
        return acc

    def saveModel(self, model, name, path='Models/'):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + name + '.pkl'
        joblib.dump(model, filename)

    def loadModel(self, name, path='Models/'):
        filename = path + name + '.pkl'
        loaded_model = joblib.load(filename)
        return loaded_model

    def getMeasurements(self, y_actual, y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == True:
                TP += 1
            if y_pred[i] == True and y_actual[i] != y_pred[i]:
                FP += 1
            if y_actual[i] == y_pred[i] == False:
                TN += 1
            if y_pred[i] == False and y_actual[i] != y_pred[i]:
                FN += 1

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        if TP + FP > 0:
            PPV = TP / (TP + FP)
        else:
            PPV = -1
        # Negative predictive value
        if TN + FN > 0:
            NPV = TN / (TN + FN)
        else:
            NPV = -1
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        if TP + FP > 0:
            FDR = FP / (TP + FP)
        else:
            FDR = -1

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        return {
            'TPR': TPR,
            'sensitivity': TPR,
            'recall': TPR,
            'TNR': TNR,
            'specificity': TNR,
            'PPV': PPV,
            'precision': PPV,
            'NPV': NPV,
            'FPR': FPR,
            'FNR': FNR,
            'FDR': FDR,
            'ACC': ACC
        }

    def GaussianProcessClassifier(self, X_train, X_test, y_train, y_test, model=None):
        if model is None:
            kernel = 1.0 * RBF(1.0)
            clf = GaussianProcessClassifier(kernel=kernel, random_state=110)
            clf.fit(X_train, y_train)
            self.gpc = clf
        else:
            clf = model
        self.gpc_predicted = clf.predict(X_test)
        self.gpc_predicted_proba = clf.predict_proba(X_test)
        self.gpc_accuracy = accuracy_score(y_test, self.gpc_predicted)
        self.gpc_log_loss = log_loss(y_test, self.gpc_predicted_proba)
        probs = self.gpc_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.gpc_auc = roc_auc_score(y_test, probs)
        self.gpc_measurements = self.getMeasurements(y_test, self.gpc_predicted)
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
        return clf

    def RandomForestClassifier(self, X_train, X_test, y_train, y_test, model=None):
        if model is None:
            clf = RandomForestClassifier(max_depth=2, random_state=110)
            clf.fit(X_train, y_train)
            self.rfc = clf
        else:
            clf = model
        self.rfc_predicted = clf.predict(X_test)
        self.rfc_predicted_proba = clf.predict_proba(X_test)
        self.rfc_accuracy = accuracy_score(y_test, self.rfc_predicted)
        self.rfc_log_loss = log_loss(y_test, self.rfc_predicted_proba)
        probs = self.rfc_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.rfc_auc = roc_auc_score(y_test, probs)
        self.rfc_measurements = self.getMeasurements(y_test, self.rfc_predicted)
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
        return clf

    def MLPClassifier(self, X_train, X_test, y_train, y_test, model=None):
        if model is None:
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 2), random_state=110)
            clf.fit(X_train, y_train)
            self.mlp = clf
        else:
            clf = model
        self.mlp_predicted = clf.predict(X_test)
        self.mlp_predicted_proba = clf.predict_proba(X_test)
        self.mlp_accuracy = accuracy_score(y_test, self.mlp_predicted)
        self.mlp_log_loss = log_loss(y_test, self.mlp_predicted_proba)
        probs = self.mlp_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.mlp_auc = roc_auc_score(y_test, probs)
        self.mlp_measurements = self.getMeasurements(y_test, self.mlp_predicted)
        if self.verbose:
            print('-MLPClassifier:')
            print("Test Accuracy: %.4f"
                  % self.mlp_accuracy)
            print("Test AUC: %.4f"
                  % self.mlp_auc)
            print("Log-loss: %.4f"
                  % self.mlp_log_loss)
        return clf

    def ComplementNB(self, X_train, X_test, y_train, y_test, model=None):
        if model is None:
            clf = ComplementNB()
            clf.fit(X_train, y_train)
            self.cnb = clf
        else:
            clf = model
        self.cnb_predicted = clf.predict(X_test)
        self.cnb_predicted_proba = clf.predict_proba(X_test)
        self.cnb_accuracy = accuracy_score(y_test, self.cnb_predicted)
        self.cnb_log_loss = log_loss(y_test, self.cnb_predicted_proba)
        probs = self.cnb_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.cnb_auc = roc_auc_score(y_test, probs)
        self.cnb_measurements = self.getMeasurements(y_test, self.cnb_predicted)
        if self.verbose:
            print('-ComplementNB:')
            print("Test Accuracy: %.4f"
                  % self.cnb_accuracy)
            print("Test AUC: %.4f"
                  % self.cnb_auc)
            print("Log-loss: %.4f"
                  % self.cnb_log_loss)
        return clf

    def GradientBoostingClassifier(self, X_train, X_test, y_train, y_test, model=None):
        if model is None:
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=110)
            clf.fit(X_train, y_train)
            self.gbc = clf
        else:
            clf = model
        self.gbc_predicted = clf.predict(X_test)
        self.gbc_predicted_proba = clf.predict_proba(X_test)
        self.gbc_accuracy = accuracy_score(y_test, self.gbc_predicted)
        self.gbc_log_loss = log_loss(y_test, self.gbc_predicted_proba)
        probs = self.gbc_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.gbc_auc = roc_auc_score(y_test, probs)
        self.gbc_measurements = self.getMeasurements(y_test, self.gbc_predicted)
        if self.verbose:
            print('-GradientBoostingClassifier:')
            print("Test Accuracy: %.4f"
                  % self.gbc_accuracy)
            print("Test AUC: %.4f"
                  % self.gbc_auc)
            print("Log-loss: %.4f"
                  % self.gbc_log_loss)
        return clf

    def NonLinearSVMClassifier(self, X_train, X_test, y_train, y_test, model=None):
        if model is None:
            clf = svm.SVC(gamma='auto', kernel='rbf', degree=3, class_weight='balanced', probability=True)
            clf.fit(X_train, y_train)
            self.nlsvm = clf
        else:
            clf = model
        self.nlsvm_predicted = clf.predict(X_test)
        self.nlsvm_predicted_proba = clf.predict_proba(X_test)
        self.nlsvm_accuracy = accuracy_score(y_test, self.nlsvm_predicted)
        self.nlsvm_log_loss = log_loss(y_test, self.nlsvm_predicted_proba)
        probs = self.nlsvm_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.nlsvm_auc = roc_auc_score(y_test, probs)
        self.nlsvm_measurements = self.getMeasurements(y_test, self.nlsvm_predicted)
        if self.verbose:
            print('-NonLinearSVMClassifier:')
            print("Test Accuracy: %.4f"
                  % self.nlsvm_accuracy)
            print("Test AUC: %.4f"
                  % self.nlsvm_auc)
            print("Log-loss: %.4f"
                  % self.nlsvm_log_loss)
        return clf

    def evaluateData(self, model, X_test, y_test):
        if model == "gpc":
            clf = self.gpc
        elif model == "rfc":
            clf = self.rfc
        elif model == "mlp":
            clf = self.mlp
        elif model == "cnb":
            clf = self.cnb
        elif model == "gbc":
            clf = self.gbc
        elif model == "nlsvm":
            clf = self.nlsvm
        else:
            return
        predicted = clf.predict(X_test)
        predicted_proba = clf.predict_proba(X_test)
        accuracy = accuracy_score(y_test, predicted)
        logloss = log_loss(y_test, predicted_proba)
        probs = predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        auc = roc_auc_score(y_test, probs)
        return predicted, predicted_proba, accuracy, auc, logloss

    def evaluateModelData(self, model, X_test, y_test):
        clf = model
        predicted = clf.predict(X_test)
        predicted_proba = clf.predict_proba(X_test)
        accuracy = accuracy_score(y_test, predicted)
        logloss = log_loss(y_test, predicted_proba)
        probs = predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        auc = roc_auc_score(y_test, probs)
        return predicted, predicted_proba, accuracy, auc, logloss


class Visualization:
    def __init__(self):
        pass

    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
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
        # im = ax.imshow(data, aspect='auto', **kwargs)
        im = sns.heatmap(data, center=0.05, square=False,
                         cbar_kws={"orientation": "horizontal", 'pad': 0.05, 'aspect': 50})

        # Create colorbar
        # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        # ax.set_yticklabels(row_labels)
        ax.set_yticklabels(row_labels, rotation=0)

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

    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts
