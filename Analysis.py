from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, log_loss, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, ShuffleSplit, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from fylearn.nfpc import FuzzyPatternClassifier
from fylearn.fpcga import FuzzyPatternClassifierGA
import matplotlib
from matplotlib import pyplot as plt
import joblib
import os
from sklearn.metrics import roc_auc_score
from sklearn import svm
import numpy as np
# from tpot import TPOTClassifier


class Model:
    def __init__(self):
        self.name = None
        self.model = None
        self.measurements = None
        self.accuracy = None
        self.auc = None
        self.log_loss = None
        self.predicted = None
        self.predicted_proba = None


class Analysis:
    def __init__(self, cv: ShuffleSplit, verbose=True):
        self.verbose = verbose
        self.cv = cv
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
        # self.tpot = None
        # self.tpot_measurements = None
        # self.tpot_accuracy = None
        # self.tpot_auc = None
        # self.tpot_log_loss = None
        # self.tpot_predicted = None
        # self.tpot_predicted_proba = None
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
        self.fpc = None
        self.fpc_measurements = None
        self.fpc_accuracy = None
        self.fpc_predicted = None
        self.fpcga = None
        self.fpcga_measurements = None
        self.fpcga_accuracy = None
        self.fpcga_predicted = None
        self.mv_measurements = None
        self.mv_accuracy = None
        self.mv_predicted = None
        self.wmv_measurements = None
        self.wmv_accuracy = None
        self.wmv_predicted = None
        self.cv_measurements = None
        self.cv_accuracy = None
        self.cv_predicted = None

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
            # 'TpotClassifier': {
            #     'measurements': self.tpot_measurements,
            #     'acc': self.tpot_accuracy,
            #     'auc': self.tpot_auc,
            #     'log_loss': self.tpot_log_loss},
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
            'FuzzyPatternClassifier': {
                'measurements': self.fpc_measurements,
                'acc': self.fpc_accuracy,
                'auc': -1,
                'log_loss': -1},
            'FuzzyPatternClassifierGA': {
                'measurements': self.fpcga_measurements,
                'acc': self.fpcga_accuracy,
                'auc': -1,
                'log_loss': -1},
            'maxVoting': {
                'measurements': self.mv_measurements,
                'acc': self.mv_accuracy,
                'auc': -1,
                'log_loss': -1},
            'weightedMaxVoting': {
                'measurements': self.wmv_measurements,
                'acc': self.wmv_accuracy,
                'auc': -1,
                'log_loss': -1},
            'customVoting': {
                'measurements': self.cv_measurements,
                'acc': self.cv_accuracy,
                'auc': -1,
                'log_loss': -1},
        }
        return acc

    def getPredictions(self):
        # pred = {
        #     'GaussianProcessClassifier': self.gpc_predicted,
        #     'RandomForestClassifier': self.rfc_predicted,
        #     'MLPClassifier': self.mlp_predicted,
        #     'ComplementNB': self.cnb_predicted,
        #     'GradientBoostingClassifier': self.gbc_predicted,
        #     'NonLinearSVMClassifier': self.nlsvm_predicted,
        # }
        pred = [
            self.gpc_predicted,
            self.rfc_predicted,
            self.mlp_predicted,
            # self.tpot_predicted,
            self.cnb_predicted,
            self.gbc_predicted,
            self.nlsvm_predicted,
            self.fpc_predicted,
            self.fpcga_predicted
        ]
        return pred

    def getClassifiersPrediction(self):
        return {
            'GaussianProcessClassifier': {'predicted': self.gpc_predicted, 'measurements': self.gpc_measurements},
            'RandomForestClassifier': {'predicted': self.rfc_predicted, 'measurements': self.rfc_measurements},
            'MLPClassifier': {'predicted': self.mlp_predicted, 'measurements': self.mlp_measurements},
            # 'TpotClassifier': {'predicted': self.tpot_predicted, 'measurements': self.tpot_measurements},
            'ComplementNB': {'predicted': self.cnb_predicted, 'measurements': self.cnb_measurements},
            'GradientBoostingClassifier': {'predicted': self.gbc_predicted, 'measurements': self.gbc_measurements},
            'NonLinearSVMClassifier': {'predicted': self.nlsvm_predicted, 'measurements': self.nlsvm_measurements},
            'FuzzyPatternClassifier': {'predicted': self.fpc_predicted, 'measurements': self.fpc_measurements},
            'FuzzyPatternClassifierGA': {'predicted': self.fpcga_predicted, 'measurements': self.fpcga_measurements}

        }

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

    # def TpotClassifier(self, X_train, X_test, y_train, y_test, model=None):
    #     if model is None:
    #         clf = TPOTClassifier(generations=5, population_size=20, cv=5,
    #                              random_state=110, verbosity=1)
    #         clf.fit(X_train, y_train)
    #         self.tpot = clf
    #     else:
    #         clf = model
    #     self.tpot_predicted = clf.predict(X_test)
    #     self.tpot_predicted_proba = clf.predict_proba(X_test)
    #     self.tpot_accuracy = accuracy_score(y_test, self.tpot_predicted)
    #     self.tpot_log_loss = log_loss(y_test, self.tpot_predicted_proba)
    #     probs = self.tpot_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
    #     self.tpot_auc = roc_auc_score(y_test, probs)
    #     self.tpot_measurements = self.getMeasurements(y_test, self.tpot_predicted)
    #     if self.verbose:
    #         print('-TpotClassifier:')
    #         print("Test Accuracy: %.4f"
    #               % self.tpot_accuracy)
    #         print("Test AUC: %.4f"
    #               % self.tpot_auc)
    #         print("Log-loss: %.4f"
    #               % self.tpot_log_loss)
    #     return clf

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
        mModel = Model()
        if model is None:
            clf = svm.SVC(gamma='auto', kernel='rbf', degree=3, class_weight='balanced', probability=True)
            clf.fit(X_train, y_train)
            self.nlsvm = clf
        else:
            clf = model
        # TODO: verify this by printing separated data of each classifier
        # It's better to use same separation of data for cv of classifiers
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        -----
        mModel.name = "NonLinearSVMClassifier"
        mModel.predicted = clf.predict(X_test)
        self.nlsvm_predicted = clf.predict(X_test)
        self.nlsvm_predicted_proba = clf.predict_proba(X_test)
        self.nlsvm_accuracy = accuracy_score(y_test, self.nlsvm_predicted)
        self.nlsvm_log_loss = log_loss(y_test, self.nlsvm_predicted_proba)
        probs = self.nlsvm_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        self.nlsvm_auc = roc_auc_score(y_test, probs)
        self.nlsvm_measurements = self.getMeasurements(y_test, self.nlsvm_predicted)
        if self.verbose:
            print('-NonLinearSVMClassifier:')
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % self.nlsvm_accuracy)
            print("Test AUC: %.4f"
                  % self.nlsvm_auc)
            print("Log-loss: %.4f"
                  % self.nlsvm_log_loss)
        return clf

    def FuzzyPatternClassifier(self, X_train, X_test, y_train, y_test, model=None):
        if model is None:
            clf = FuzzyPatternClassifier()
            clf.fit(X_train, y_train)
            self.fpc = clf
        else:
            clf = model
        self.fpc_predicted = clf.predict(X_test)
        # self.nlsvm_predicted_proba = clf.predict_proba(X_test)
        self.fpc_accuracy = accuracy_score(y_test, self.fpc_predicted)
        # self.nlsvm_log_loss = log_loss(y_test, self.nlsvm_predicted_proba)
        # probs = self.nlsvm_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # self.nlsvm_auc = roc_auc_score(y_test, probs)
        self.fpc_measurements = self.getMeasurements(y_test, self.fpc_predicted)
        if self.verbose:
            print('-FuzzyPatternClassifier:')
            print("Test Accuracy: %.4f"
                  % self.fpc_accuracy)
            # print("Test AUC: %.4f"
            #       % self.nlsvm_auc)
            # print("Log-loss: %.4f"
            #       % self.nlsvm_log_loss)
        return clf

    def FuzzyPatternClassifierGA(self, X_train, X_test, y_train, y_test, model=None):
        if model is None:
            clf = FuzzyPatternClassifierGA()
            clf.fit(X_train, y_train)
            self.fpcga = clf
        else:
            clf = model
        self.fpcga_predicted = clf.predict(X_test)
        # self.nlsvm_predicted_proba = clf.predict_proba(X_test)
        self.fpcga_accuracy = accuracy_score(y_test, self.fpcga_predicted)
        # self.nlsvm_log_loss = log_loss(y_test, self.nlsvm_predicted_proba)
        # probs = self.nlsvm_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # self.nlsvm_auc = roc_auc_score(y_test, probs)
        self.fpcga_measurements = self.getMeasurements(y_test, self.fpcga_predicted)
        if self.verbose:
            print('-FuzzyPatternClassifierGA:')
            print("Test Accuracy: %.4f"
                  % self.fpcga_accuracy)
            # print("Test AUC: %.4f"
            #       % self.nlsvm_auc)
            # print("Log-loss: %.4f"
            #       % self.nlsvm_log_loss)
        return clf

    def maxVoting(self, y_test):
        result = []
        length = 0
        for i in self.getPredictions():
            if i is not None:
                length = len(i)
                break
        if length == 0:
            raise ValueError('All classifiers predictions are None')
        for idx in range(length):
            preds = []
            for classifier in self.getPredictions():
                if classifier is not None:
                    preds.append(classifier[idx])
            result.append(np.argmax(np.bincount(preds)) > 0)
            # print(preds, np.argmax(np.bincount(preds)) > 0)
        self.mv_predicted = result
        # self.mv_predicted_proba = clf.predict_proba(X_test)
        self.mv_accuracy = accuracy_score(y_test, self.mv_predicted)
        # self.mv_log_loss = log_loss(y_test, self.mv_predicted_proba)
        # probs = self.mv_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # self.mv_auc = roc_auc_score(y_test, probs)
        self.mv_measurements = self.getMeasurements(y_test, self.mv_predicted)
        if self.verbose:
            print('-maxVoting:')
            print("Test Accuracy: %.4f"
                  % self.mv_accuracy)
            # print("Test AUC: %.4f"
            #       % self.nlsvm_auc)
            # print("Log-loss: %.4f"
            #       % self.nlsvm_log_loss)

    def weightedMaxVoting(self, y_test, weights):
        result = []
        length = 0
        for i in self.getPredictions():
            if i is not None:
                length = len(i)
                break
        if length == 0:
            raise ValueError('All classifiers predictions are None')
        for idx in range(length):
            preds = []
            for classifier in self.getPredictions():
                if classifier is not None:
                    preds.append(classifier[idx])
            result.append(np.average(preds, weights=weights) > 0.5)
            # result.append(np.argmax(np.bincount(preds)) > 0)
            # print(preds, np.average(preds, weights=weights))
        self.wmv_predicted = result
        # self.mv_predicted_proba = clf.predict_proba(X_test)
        self.wmv_accuracy = accuracy_score(y_test, self.wmv_predicted)
        # self.mv_log_loss = log_loss(y_test, self.mv_predicted_proba)
        # probs = self.mv_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # self.mv_auc = roc_auc_score(y_test, probs)
        self.wmv_measurements = self.getMeasurements(y_test, self.wmv_predicted)
        if self.verbose:
            print('-weightedMaxVoting:')
            print("Test Accuracy: %.4f"
                  % self.wmv_accuracy)
            # print("Test AUC: %.4f"
            #       % self.nlsvm_auc)
            # print("Log-loss: %.4f"
            #       % self.nlsvm_log_loss)

    def customVoting(self, y_test, size=2, threshold_tpr=.7, threshold_tnr=.7):
        result = []
        length = 0
        for i in self.getPredictions():
            if i is not None:
                length = len(i)
                break
        if length == 0:
            raise ValueError('All classifiers predictions are None')
        for sample_idx in range(length):
            # preds = []
            predictions = []
            for classifier, value in self.getClassifiersPrediction().items():
                predicted = value['predicted']
                if predicted is not None:
                    measurements = value['measurements']
                    sensitivity = measurements['sensitivity']
                    specificity = measurements['specificity']
                    # print(classifier, predicted[sample_idx], sensitivity, specificity)
                    predictions.append({
                        'prediction': predicted[sample_idx],
                        'TPR': sensitivity,
                        'TNR': specificity
                    })
            predictions_tpr_sorted = sorted(predictions, key=lambda k: k['TPR'], reverse=True)
            predictions_tnr_sorted = sorted(predictions, key=lambda k: k['TNR'], reverse=True)
            sum = 0
            count = 0
            for m in range(size):
                if predictions_tpr_sorted[m]['TPR'] >= threshold_tpr and \
                        predictions_tnr_sorted[m]['TNR'] >= threshold_tnr:
                    if predictions_tpr_sorted[m]['prediction']:
                        sum += predictions_tpr_sorted[m]['TPR']
                    else:
                        sum -= predictions_tpr_sorted[m]['TNR']
                    if predictions_tnr_sorted[m]['prediction']:
                        sum += predictions_tnr_sorted[m]['TPR']
                    else:
                        sum -= predictions_tnr_sorted[m]['TNR']
                    count += 1
                # if predictions_tpr_sorted[m]['TPR'] >= threshold_tpr:
                #     if predictions_tpr_sorted[m]['prediction']:
                #         sum += predictions_tpr_sorted[m]['TPR']
                #     else:
                #         sum -= predictions_tpr_sorted[m]['TNR']
                # if predictions_tnr_sorted[m]['TNR'] >= threshold_tnr:
                #     if predictions_tnr_sorted[m]['prediction']:
                #         sum += predictions_tnr_sorted[m]['TPR']
                #     else:
                #         sum -= predictions_tnr_sorted[m]['TNR']
            sum /= count
            result.append(sum >= 1)
            # if result[sample_idx] != y_test[sample_idx]:
            #     print('predicted:', result[sample_idx], 'sum:', sum, 'actual:', y_test[sample_idx], '\n')
            # else:
            #     print('OK, sum:', sum, '\n')
        self.cv_predicted = result
        # self.mv_predicted_proba = clf.predict_proba(X_test)
        self.cv_accuracy = accuracy_score(y_test, self.cv_predicted)
        # self.mv_log_loss = log_loss(y_test, self.mv_predicted_proba)
        # probs = self.mv_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # self.mv_auc = roc_auc_score(y_test, probs)
        self.cv_measurements = self.getMeasurements(y_test, self.cv_predicted)
        if self.verbose:
            print('-customVoting:')
            print("Test Accuracy: %.4f"
                  % self.cv_accuracy)
            # print("Test AUC: %.4f"
            #       % self.nlsvm_auc)
            # print("Log-loss: %.4f"
            #       % self.nlsvm_log_loss)

    # def owa(self, X_train, X_test, y_train, y_test, num_args, lr=0.9, epoch_num=150):
    #     # Initialize
    #     landa = np.random.rand(num_args)
    #     # landa = np.zeros(num_args)
    #     w = np.ones(num_args) * (1.0 / num_args)
    #     d_estimate = np.sum([np.exp(landa[i]) / np.sum(np.exp(landa)) for i in range(num_args)])
    #     # d_estimate = 0
    #
    #     # Train
    #     for epoch in range(epoch_num):
    #         lr /= 1.001
    #         for idx, sample in enumerate(X_train):
    #             b = np.sort(sample)[::-1]
    #             diff = w * (b - d_estimate) * (d_estimate - y_train[idx])
    #             landa -= lr * diff
    #             w = [np.exp(landa[i]) / np.sum(np.exp(landa)) if np.exp(landa[i]) / np.sum(
    #                 np.exp(landa)) > 1e-5 else 0
    #                  for
    #                  i in range(num_args)]
    #             # print(w)
    #             d_estimate = np.sum(b * w)
    #
    #     # print('\nLambdas:', np.round(landa, 2))
    #     # print('Weights:', np.round(w, 2))
    #     # print('\nSample\t\tAggregated Value\tEstimated Value')
    #     # print('-----------------------------------------------')
    #     # for idx, sample in enumerate(train_data):
    #     #     print('Sample', (idx + 1), '\t\t', train_label[idx], '\t\t\t', np.round(np.sum(np.sort(train_data[idx])[::-1] * w), 2))
    #     # Prediction
    #     self.owa_predicted = []
    #     for idx, sample in enumerate(X_test):
    #         b = np.sort(sample)[::-1]
    #         d_estimate = np.sum(b * w)
    #         self.owa_predicted.append(d_estimate)
    #     # self.nlsvm_predicted_proba = clf.predict_proba(X_test)
    #     self.nlsvm_accuracy = accuracy_score(y_test, self.nlsvm_predicted)
    #     self.nlsvm_log_loss = log_loss(y_test, self.nlsvm_predicted_proba)
    #     probs = self.nlsvm_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
    #     self.nlsvm_auc = roc_auc_score(y_test, probs)
    #     self.nlsvm_measurements = self.getMeasurements(y_test, self.nlsvm_predicted)
    #     if self.verbose:
    #         print('-NonLinearSVMClassifier:')
    #         print("Test Accuracy: %.4f"
    #               % self.nlsvm_accuracy)
    #         print("Test AUC: %.4f"
    #               % self.nlsvm_auc)
    #         print("Log-loss: %.4f"
    #               % self.nlsvm_log_loss)
    #     return clf

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

    # evaluate a model
    # TODO
    def evaluateModel(self, X, y, model):
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        # print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
        return scores
