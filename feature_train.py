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
from sklearn import manifold
from sklearn.decomposition import PCA



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

        
        
def train_hog():

    feat_path = "data/train_feat_hog.npy"
    label_path = "data/train_label_hog.npy"

    x_train = np.load(feat_path)
    y_train = np.load(label_path)
    
    feat_path = "data/valid_feat_hog.npy"
    label_path = "data/valid_label_hog.npy"

    x_valid = np.load(feat_path)
    y_valid = np.load(label_path)
    
    pca = PCA(n_components=100)
    
    print(x_train.shape, y_train.shape)
    train_pca = pca.fit_transform(x_train)
    valid_pca = pca.transform(x_valid)
    
    model = GradientBoostingClassifier(random_state=9)
    
    kfold = KFold(n_splits=3, random_state=7)
    cv_results = cross_val_score(
        model,
        train_pca,
        y_train,
        cv=kfold,
        scoring="accuracy",  # "roc_auc",  #
    )
    msg = "%s: %f (%f)" % ('model', cv_results.mean(), cv_results.std())
    
    model.fit(train_pca, y_train)
    
    # print(classification_report(y_test, y_pred))

    print(train_pca.shape, msg)
    print(confusion_matrix(y_valid, model.predict(valid_pca)))

        
def train_diff_feature(df_train, df_test):
    
    
    x_train, y_train = train_data(df_train, replace=False)
    x_valid, y_valid = valid_data(replace=False)
    
    train_histogram = x_train[:, 0:512]
    train_haralick = x_train[:, 512:525]
    train_hu_moments = x_train[:, 525:532]
    train_lbp = x_train[:, 532:558]
    train_sift = x_train[:, 558:]
    
    valid_histogram = x_valid[:, 0:512]
    valid_haralick = x_valid[:, 512:525]
    valid_hu_moments = x_valid[:, 525:532]
    valid_lbp = x_valid[:, 532:558]
    valid_sift = x_valid[:, 558:]
    
    train_features = [train_histogram, train_haralick, train_hu_moments, train_lbp, train_sift]
    valid_features = [valid_histogram, valid_haralick, valid_hu_moments, valid_lbp, valid_sift]
    
    pca = PCA(n_components=20)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    
    # all_hist = np.vstack([train_histogram, train_histogram])
    # all_tsne = tsne.fit_transform(all_hist)
    
    # train_tsne = all_tsne[:train_histogram.shape[0]]
    # valid_tsne = all_tsne[train_histogram.shape[0]:]
    
    train_pca = pca.fit_transform(train_histogram)
    valid_pca = pca.transform(valid_histogram)
    
    train_features = [train_histogram, train_pca]
    valid_features = [valid_histogram, valid_pca]
    
    for train_feature, valid_feature in zip(train_features, valid_features) :
    
        model = GradientBoostingClassifier(random_state=9)
        
        kfold = KFold(n_splits=3, random_state=7)
        cv_results = cross_val_score(
            model,
            train_feature,
            y_train,
            cv=kfold,
            scoring="accuracy",  # "roc_auc",  #
        )
        msg = "%s: %f (%f)" % ('model', cv_results.mean(), cv_results.std())
        print(train_feature.shape, msg)
        model.fit(train_feature, y_train)
        print(confusion_matrix(y_valid, model.predict(valid_feature)))
        
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

    # models, preds = train_main(df_train, df_test)
    
    # for model in models:
        # pickle.dump(model[1], open('log/' + model[0].lower(), "wb"))
    
    # train_diff_feature(df_train, df_test)
    train_hog()