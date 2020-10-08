from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold
from sklearn.linear_model import LogisticRegression


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
