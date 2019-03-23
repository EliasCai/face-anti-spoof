# -- coding:utf-8 --

import os, glob, sys
from tqdm import tqdm
import pickle
import numpy as np

from create_index_face import readIndex, readDirect

from feature_svm import train_data, test_data, valid_data, ExtractFeature

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class AntiModel:
    def __init__(
        self,
        kmean_path="log/kmean.sav",
        scaler_path="log/scaler.sav",
        cart_path="log/cart.sav",
        rf_path="log/rf.sav",
        gbdt_path="log/gbdt.sav",
    ):

        CART = pickle.load(open(cart_path, "rb"))
        RF = pickle.load(open(rf_path, "rb"))
        GBDT = pickle.load(open(gbdt_path, "rb"))
        self.models = [RF, GBDT]
        self.et = ExtractFeature(kmean_path=kmean_path, scaler_path=scaler_path)

    def predict_path(self, path, threshold=0.5):
        feature = self.et.extract_features(path)
        preds = []
        for model in self.models:
            pred = model.predict_proba(feature[:].reshape(1,-1))
            preds.append(pred[:,1:])
        preds = np.hstack(preds)
        return preds.min() > threshold


def train_main(df_train, df_test):

    num_trees = 500
    models = []
    preds = []
    x_train, y_train = train_data(df_train, replace=False)
    x_test, y_test = test_data(df_test, replace=False)
    x_valid, y_valid = valid_data(replace=False)

    # models.append(("LR", LogisticRegression(random_state=9, solver="lbfgs")))
    # models.append(("LDA", LinearDiscriminantAnalysis()))
    # models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier(random_state=9)))
    models.append(
        ("RF", RandomForestClassifier(n_estimators=num_trees, random_state=9))
    )
    models.append(("GBDT", GradientBoostingClassifier(random_state=9)))
    # models.append(("NB", GaussianNB()))
    # models.append(("SVM", SVC(random_state=9, gamma="auto")))
    for name, model in models:
        kfold = KFold(n_splits=3, random_state=7)
        cv_results = cross_val_score(
            model,
            x_train[:, :],
            y_train,
            cv=kfold,
            scoring="accuracy",  # "roc_auc",  #
        )

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

        model.fit(x_train[:, :], y_train)
        print(confusion_matrix(y_test, model.predict(x_test[:, :])))
        print(confusion_matrix(y_valid, model.predict(x_valid[:, :])))

        preds.append(model.predict_proba(x_valid[:, :])[:, 1])
    return models, preds


if __name__ == "__main__":

    df_train, df_test = readIndex()

    models, preds = train_main(df_train, df_test)
    
    for model in models:
        pickle.dump(model[1], open('log/' + model[0].lower(), "wb"))
