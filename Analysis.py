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
from datetime import datetime


# from tpot import TPOTClassifier


class Model:
    def __init__(self):
        self.name: str = None
        self.model = None
        self.measurements: dict = None
        self.accuracy: float = None
        self.auc: float = None
        self.log_loss: float = None
        self.predicted = None
        self.predicted_proba = None
        self.cv_scores = None
        self.time: float = None  # milliseconds


class Analysis:
    def __init__(self, cv: ShuffleSplit, verbose=True):
        self.verbose = verbose
        self.cv = cv
        self.gpc = None
        self.rfc = None
        self.mlp = None
        self.cnb = None
        self.gbc = None
        self.nlsvm = None
        self.fpc = None
        self.fpcga = None
        self.mv = None
        self.wmv = None
        self.custom_voting = None

    def getModels(self):
        models = [self.gpc,
                  self.rfc,
                  self.mlp,
                  self.cnb,
                  self.gbc,
                  self.nlsvm,
                  self.fpc,
                  self.fpcga,
                  self.mv,
                  self.wmv,
                  self.custom_voting]
        return models

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
        acc = {}
        for model in self.getModels():
            if model is not None:
                accuracy = round(model.accuracy, 4) * 100
                auc = None
                tpr = round(model.measurements['TPR'], 4) * 100
                tnr = round(model.measurements['TNR'], 4) * 100
                ppv = round(model.measurements['PPV'], 4) * 100
                log_loss = None
                cv_test_score_mean = None
                cv_test_score_std = None
                cv_fit_time = None
                time_test = model.time
                if model.auc is not None:
                    auc = round(model.auc, 4) * 100
                if model.log_loss is not None:
                    log_loss = round(model.log_loss, 2)
                if model.cv_scores is not None:
                    cv_test_score_mean = round(model.cv_scores['test_score'].mean(), 4) * 100
                    cv_test_score_std = round(model.cv_scores['test_score'].std() * 2, 4) * 100  # 95% Confidence Interval
                    cv_fit_time = round(model.cv_scores['fit_time'].mean(), 4)
                #     print(model.cv_scores)
                # print(cv_test_score_mean)
                # print(cv_test_score_std)
                # print(cv_fit_time)
                acc[model.name] = {
                    'Acc': accuracy,
                    'AUC': auc,
                    'TPR': tpr,
                    'TNR': tnr,
                    'PPV': ppv,
                    'log loss': log_loss,
                    'cv_test_score_mean': cv_test_score_mean,
                    'cv_test_score_std': cv_test_score_std,
                    'cv_fit_time': cv_fit_time,
                    'Test Time(ms)': time_test
                }
        return acc

    def getPredictions(self):
        pred = []
        for model in self.getModels():
            if model is not None:
                pred.append(model.predicted)
        # pred = [
        #     self.gpc_predicted,
        #     self.rfc_predicted,
        #     self.mlp_predicted,
        #     # self.tpot_predicted,
        #     self.cnb_predicted,
        #     self.gbc_predicted,
        #     self.nlsvm_predicted,
        #     self.fpc_predicted,
        #     self.fpcga_predicted
        # ]
        return pred

    def getClassifiersPrediction(self):
        pred = {}
        for model in self.getModels():
            if model is not None:
                pred[model.name] = {
                    'predicted': model.predicted,
                    'measurements': model.measurements
                }
        return pred

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
        mModel = Model()
        if model is None:
            kernel = 1.0 * RBF(1.0)
            clf = GaussianProcessClassifier(kernel=kernel, random_state=110)
            clf.fit(X_train, y_train)
            mModel.model = clf
        else:
            clf = model
        mModel.name = "Gaussian Process"
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        start_time = datetime.now()
        mModel.predicted = clf.predict(X_test)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        mModel.log_loss = log_loss(y_test, mModel.predicted_proba)
        probs = mModel.predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        mModel.auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        mModel.cv_scores = scores
        self.gpc = mModel
        if self.verbose:
            print(mModel.name)
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            print("Test AUC: %.4f"
                  % mModel.auc)
            print("Log-loss: %.4f"
                  % mModel.log_loss)
        return clf

    def RandomForestClassifier(self, X_train, X_test, y_train, y_test, model=None):
        mModel = Model()
        if model is None:
            clf = RandomForestClassifier(max_depth=2, random_state=110)
            clf.fit(X_train, y_train)
            mModel.model = clf
        else:
            clf = model
        mModel.name = "Random Forest"
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        start_time = datetime.now()
        mModel.predicted = clf.predict(X_test)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        mModel.log_loss = log_loss(y_test, mModel.predicted_proba)
        probs = mModel.predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        mModel.auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        mModel.cv_scores = scores
        self.rfc = mModel
        if self.verbose:
            print(mModel.name)
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            print("Test AUC: %.4f"
                  % mModel.auc)
            print("Log-loss: %.4f"
                  % mModel.log_loss)

        # fpr, tpr, thresholds = roc_curve(y_test, probs)
        # self.plot_roc_curve(fpr, tpr, self.rfc_auc)
        return clf

    def MLPClassifier(self, X_train, X_test, y_train, y_test, model=None):
        mModel = Model()
        if model is None:
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 2), random_state=110)
            clf.fit(X_train, y_train)
            mModel.model = clf
        else:
            clf = model
        mModel.name = "MLP"
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        start_time = datetime.now()
        mModel.predicted = clf.predict(X_test)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        mModel.log_loss = log_loss(y_test, mModel.predicted_proba)
        probs = mModel.predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        mModel.auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        mModel.cv_scores = scores
        self.mlp = mModel
        if self.verbose:
            print(mModel.name)
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            print("Test AUC: %.4f"
                  % mModel.auc)
            print("Log-loss: %.4f"
                  % mModel.log_loss)
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
        mModel = Model()
        if model is None:
            clf = ComplementNB()
            clf.fit(X_train, y_train)
            mModel.model = clf
        else:
            clf = model
        mModel.name = "Complement NB"
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        start_time = datetime.now()
        mModel.predicted = clf.predict(X_test)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        mModel.log_loss = log_loss(y_test, mModel.predicted_proba)
        probs = mModel.predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        mModel.auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        mModel.cv_scores = scores
        self.cnb = mModel
        if self.verbose:
            print(mModel.name)
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            print("Test AUC: %.4f"
                  % mModel.auc)
            print("Log-loss: %.4f"
                  % mModel.log_loss)
        return clf

    def GradientBoostingClassifier(self, X_train, X_test, y_train, y_test, model=None):
        mModel = Model()
        if model is None:
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=110)
            clf.fit(X_train, y_train)
            mModel.model = clf
        else:
            clf = model
        mModel.name = "Gradient Boosting"
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        start_time = datetime.now()
        mModel.predicted = clf.predict(X_test)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        mModel.log_loss = log_loss(y_test, mModel.predicted_proba)
        probs = mModel.predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        mModel.auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        mModel.cv_scores = scores
        self.gbc = mModel
        if self.verbose:
            print(mModel.name)
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            print("Test AUC: %.4f"
                  % mModel.auc)
            print("Log-loss: %.4f"
                  % mModel.log_loss)
        return clf

    def NonLinearSVMClassifier(self, X_train, X_test, y_train, y_test, model=None):
        mModel = Model()
        if model is None:
            clf = svm.SVC(gamma='auto', kernel='rbf', degree=3, class_weight='balanced', probability=True)
            clf.fit(X_train, y_train)
            mModel.model = clf
        else:
            clf = model
        # TODO: verify this by printing separated data of each classifier
        # It's better to use same separation of data for cv of classifiers
        mModel.name = "Non Linear SVM"
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        start_time = datetime.now()
        mModel.predicted = clf.predict(X_test)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        mModel.log_loss = log_loss(y_test, mModel.predicted_proba)
        probs = mModel.predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        mModel.auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        mModel.cv_scores = scores
        self.nlsvm = mModel
        if self.verbose:
            print(mModel.name)
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            print("Test AUC: %.4f"
                  % mModel.auc)
            print("Log-loss: %.4f"
                  % mModel.log_loss)
        return clf

    def FuzzyPatternClassifier(self, X_train, X_test, y_train, y_test, model=None):
        mModel = Model()
        if model is None:
            clf = FuzzyPatternClassifier()
            clf.fit(X_train, y_train)
            mModel.model = clf
        else:
            clf = model
        mModel.name = "Fuzzy Pattern"
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        start_time = datetime.now()
        mModel.predicted = clf.predict(X_test)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        # mModel.predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        # mModel.log_loss = log_loss(y_test, mModel.predicted_proba)
        # probs = mModel.predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # mModel.auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        mModel.cv_scores = scores
        self.fpc = mModel
        if self.verbose:
            print(mModel.name)
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            # print("Test AUC: %.4f"
            #       % mModel.auc)
            # print("Log-loss: %.4f"
            #       % mModel.log_loss)
        return clf

    def FuzzyPatternClassifierGA(self, X_train, X_test, y_train, y_test, model=None):
        mModel = Model()
        if model is None:
            clf = FuzzyPatternClassifierGA()
            clf.fit(X_train, y_train)
            mModel.model = clf
        else:
            clf = model
        mModel.name = "Fuzzy Pattern GA"
        scores = cross_validate(clf, X_train, y_train, cv=self.cv)  # ['test_score', 'fit_time', 'score_time']
        start_time = datetime.now()
        mModel.predicted = clf.predict(X_test)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        # mModel.predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        # mModel.log_loss = log_loss(y_test, mModel.predicted_proba)
        # probs = mModel.predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # mModel.auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        mModel.cv_scores = scores
        self.fpcga = mModel
        if self.verbose:
            print(mModel.name)
            print("Accuracy (.95 CI): %0.4f (+/- %0.4f)"
                  % (scores['test_score'].mean(), scores['test_score'].std() * 2))
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            # print("Test AUC: %.4f"
            #       % mModel.auc)
            # print("Log-loss: %.4f"
            #       % mModel.log_loss)
        return clf

    def maxVoting(self, y_test):
        mModel = Model()
        result = []
        length = 0
        for i in self.getPredictions():
            if i is not None:
                length = len(i)
                break
        if length == 0:
            raise ValueError('All classifiers predictions are None')
        start_time = datetime.now()
        for idx in range(length):
            preds = []
            for classifier in self.getPredictions():
                if classifier is not None:
                    preds.append(classifier[idx])
            result.append(np.argmax(np.bincount(preds)) > 0)
            # print(preds, np.argmax(np.bincount(preds)) > 0)
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.name = "MaxVoting"
        mModel.predicted = result
        # self.mv_predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        # self.mv_log_loss = log_loss(y_test, self.mv_predicted_proba)
        # probs = self.mv_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # self.mv_auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        self.mv = mModel
        if self.verbose:
            print(mModel.name)
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            # print("Test AUC: %.4f"
            #       % self.nlsvm_auc)
            # print("Log-loss: %.4f"
            #       % self.nlsvm_log_loss)

    def weightedMaxVoting(self, y_test, weights):
        mModel = Model()
        result = []
        length = 0
        for i in self.getPredictions():
            if i is not None:
                length = len(i)
                break
        if length == 0:
            raise ValueError('All classifiers predictions are None')
        start_time = datetime.now()
        for idx in range(length):
            preds = []
            for classifier in self.getPredictions():
                if classifier is not None:
                    preds.append(classifier[idx])
            result.append(np.average(preds, weights=weights) > 0.5)
            # result.append(np.argmax(np.bincount(preds)) > 0)
            # print(preds, np.average(preds, weights=weights))
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.name = "Weighted Max Voting"
        mModel.predicted = result
        # self.mv_predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        # self.mv_log_loss = log_loss(y_test, self.mv_predicted_proba)
        # probs = self.mv_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # self.mv_auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        self.wmv = mModel
        if self.verbose:
            print(mModel.name)
            print("Test Accuracy: %.4f"
                  % self.wmv_accuracy)
            # print("Test AUC: %.4f"
            #       % self.nlsvm_auc)
            # print("Log-loss: %.4f"
            #       % self.nlsvm_log_loss)

    def customVoting(self, y_test, size=2, threshold_tpr=.7, threshold_tnr=.7):
        mModel = Model()
        result = []
        length = 0
        for i in self.getPredictions():
            if i is not None:
                length = len(i)
                break
        if length == 0:
            raise ValueError('All classifiers predictions are None')
        start_time = datetime.now()
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
        mModel.time = round((datetime.now() - start_time).total_seconds() * 1000, 3)
        mModel.name = "Custom Voting"
        mModel.predicted = result
        # self.mv_predicted_proba = clf.predict_proba(X_test)
        mModel.accuracy = accuracy_score(y_test, mModel.predicted)
        # self.mv_log_loss = log_loss(y_test, self.mv_predicted_proba)
        # probs = self.mv_predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
        # self.mv_auc = roc_auc_score(y_test, probs)
        mModel.measurements = self.getMeasurements(y_test, mModel.predicted)
        self.custom_voting = mModel
        if self.verbose:
            print(mModel.name)
            print("Test Accuracy: %.4f"
                  % mModel.accuracy)
            # print("Test AUC: %.4f"
            #       % self.nlsvm_auc)
            # print("Log-loss: %.4f"
            #       % self.nlsvm_log_loss)

    def owa(self, X_train, X_test, y_train, y_test, num_args, lr=0.9, epoch_num=150):
        # Initialize
        landa = np.random.rand(num_args)
        # landa = np.zeros(num_args)
        w = np.ones(num_args) * (1.0 / num_args)
        d_estimate = np.sum([np.exp(landa[i]) / np.sum(np.exp(landa)) for i in range(num_args)])
        # d_estimate = 0

        # Train
        for epoch in range(epoch_num):
            lr /= 1.001
            for idx, sample in enumerate(X_train):
                b = np.sort(sample)[::-1]
                diff = w * (b - d_estimate) * (d_estimate - y_train[idx])
                landa -= lr * diff
                w = [np.exp(landa[i]) / np.sum(np.exp(landa)) if np.exp(landa[i]) / np.sum(
                    np.exp(landa)) > 1e-5 else 0
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
        self.owa_predicted = []
        for idx, sample in enumerate(X_test):
            b = np.sort(sample)[::-1]
            d_estimate = np.sum(b * w)
            self.owa_predicted.append(d_estimate)
        self.owa_accuracy = accuracy_score(y_test, self.owa_predicted)
        self.owa_measurements = self.getMeasurements(y_test, self.owa_predicted)
        if self.verbose:
            print('-OWA:')
            print("Test Accuracy: %.4f"
                  % self.owa_accuracy)

    # def evaluateData(self, model, X_test, y_test):
    #     if model == "gpc":
    #         clf = self.gpc
    #     elif model == "rfc":
    #         clf = self.rfc
    #     elif model == "mlp":
    #         clf = self.mlp
    #     elif model == "cnb":
    #         clf = self.cnb
    #     elif model == "gbc":
    #         clf = self.gbc
    #     elif model == "nlsvm":
    #         clf = self.nlsvm
    #     else:
    #         return
    #     predicted = clf.predict(X_test)
    #     predicted_proba = clf.predict_proba(X_test)
    #     accuracy = accuracy_score(y_test, predicted)
    #     logloss = log_loss(y_test, predicted_proba)
    #     probs = predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
    #     auc = roc_auc_score(y_test, probs)
    #     return predicted, predicted_proba, accuracy, auc, logloss
    #
    # def evaluateModelData(self, model, X_test, y_test):
    #     clf = model
    #     predicted = clf.predict(X_test)
    #     predicted_proba = clf.predict_proba(X_test)
    #     accuracy = accuracy_score(y_test, predicted)
    #     logloss = log_loss(y_test, predicted_proba)
    #     probs = predicted_proba[:, 1]  # Keep Probabilities of the positive class only.
    #     auc = roc_auc_score(y_test, probs)
    #     return predicted, predicted_proba, accuracy, auc, logloss
    #
    # # evaluate a model
    # # TODO
    # def evaluateModel(self, X, y, model):
    #     # define evaluation procedure
    #     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    #     # evaluate model
    #     scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    #     # print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    #     return scores
