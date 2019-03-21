# -- coding:utf-8 --

import os, glob, sys
from tqdm import tqdm

from create_index_face import readIndex
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


from create_index_face import readDirect
from multiprocessing import Pool
from sklearn.externals import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import mahotas
import cv2
from skimage import feature
import pickle


# fixed-sizes for image
fixed_size = tuple((300, 300))

# no.of.trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same results
seed = 9

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist(
        [image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]
    )
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# feature-descriptor-4: LBP
def fd_lbp(image, numPoints=24, radius=8, eps=1e-7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(
        lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2)
    )

    # normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum() + eps
    return hist


def parallel_cluster_feat(df):

    features, labels = [], []
    sift_object = cv2.xfeatures2d.SIFT_create()

    for rowid, row in tqdm(df.iterrows()):

        image = cv2.imread(row["path"], 0)
        image = cv2.resize(image, fixed_size)
        # kp, des = self.im_helper.features(image)
        kp, des = sift_object.detectAndCompute(image, None)
        # print('des', des.shape)
        features.append(des)
        labels.append(row["label"])

    # features = np.vstack(features)
    # features = formatND(features)
    # print(features.shape)
    labels = np.array(labels).reshape((-1, 1))

    return features, labels


def formatND(l):
    """    
        restructures list into vstack array of shape
        M samples x N features for sklearn

        """
    vStack = np.array(l[0])
    for remaining in l[1:]:
        vStack = np.vstack((vStack, remaining))
    # self.descriptor_vstack = vStack.copy()
    return vStack


def cp_img(df):
    pass


class TrainModel:

    def __init__(self, no_clusters=50):

        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
    
    
    def extract_features(self, img_path):
    
        image = cv2.imread(img_path)
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = fd_hu_moments(image)
        # print('hu', fv_hu_moments.shape)
        # print(fv_hu_moments.shape)
        fv_haralick = fd_haralick(image)
        # print('hara', fv_haralick.shape)
        fv_histogram = fd_histogram(image)
        # print(q'hist', fv_histogram.shape)
        fv_lbp = fd_lbp(image)
        fv_cluster = self.fd_cluster2(image)
        # print('cluster', fv_cluster.shape)
        feature = np.hstack(
            [fv_histogram, fv_haralick, fv_hu_moments, fv_lbp, fv_cluster]
        )
        return feature
    
    def generate_features(self, df):

        features, labels = [], []
        # train_labels, test_labels = [], []

        for rowid, row in tqdm(df.iterrows()):
            
            
            img_path = row["path"]
            
            feature = self.extract_features(img_path)
            
            # print('all', feature.shape)
            # update the list of labels and feature vectors
            labels.append(row["label"])
            features.append(feature)

        features = np.vstack(features)
        # fv_cluster = self.fd_cluster(df)

        # features = np.hstack([features, fv_cluster])  #

        labels = np.array(labels).reshape((-1, 1))

        return features, labels

    def fd_cluster2(self, image):

        sift_object = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = sift_object.detectAndCompute(gray, None)
        # print('des', des.shape)
        feature = np.array(des)
        kmeans_ret = self.bov_helper.cluster(feature)
        mega_histogram = self.bov_helper.developVocabulary(
            n_images=1, descriptor_list=[feature]
        )

        mega_histogram = self.bov_helper.standardize(raw=mega_histogram)

        return mega_histogram.ravel()

    def fd_cluster(self, df, poolNum=20, splitNum=20):
        df_split = np.array_split(df, splitNum)
        features = []

        with Pool(poolNum) as pool:
            result = list(
                tqdm(pool.imap(parallel_cluster_feat, df_split), total=splitNum)
            )
        # with joblib.Parallel(n_jobs=poolNum, verbose=1) as parallel:
        # result = parallel(joblib.delayed(self.parallel_cluster_feat)(s) for s in df_split)
        for r in result:
            features += r[0]
        features_np = formatND(features)
        labels = np.vstack([r[1] for r in result])  #

        self.bov_helper.cluster(features_np)
        self.bov_helper.developVocabulary(
            n_images=len(features), descriptor_list=features
        )

        self.bov_helper.standardize()

        return self.bov_helper.mega_histogram  # , labels

    def train_cluster(self, df, poolNum=20, splitNum=20, replace=False):

        if replace:
            self.bov_helper.kmeans_obj = None
            self.bov_helper.scale = None

        df_split = np.array_split(df, splitNum)
        features = []

        with Pool(poolNum) as pool:
            result = list(
                tqdm(pool.imap(parallel_cluster_feat, df_split), total=splitNum)
            )
        # with joblib.Parallel(n_jobs=poolNum, verbose=1) as parallel:
        # result = parallel(joblib.delayed(self.parallel_cluster_feat)(s) for s in df_split)
        for r in result:
            features += r[0]
        features_np = formatND(features)
        labels = np.vstack([r[1] for r in result])  #

        self.bov_helper.cluster(features_np)
        self.bov_helper.developVocabulary(
            n_images=len(features), descriptor_list=features
        )

        self.bov_helper.standardize()
        self.train_clf(self.bov_helper.mega_histogram, labels)
        # return features, features_np, labels

    def train_clf(self, x, y):
        """
        uses sklearn.svm.SVC classifier (SVM) 


        """
        # print("Training SVM")
        # print(self.clf)
        # print("Train labels", train_labels)
        # self.clf.fit(self.mega_histogram, train_labels)

        models = []
        results = []
        names = []
        scoring = "accuracy"
        models.append(("LR", LogisticRegression(random_state=9, solver="lbfgs")))
        # models.append(("LDA", LinearDiscriminantAnalysis()))
        # models.append(("KNN", KNeighborsClassifier()))
        models.append(("CART", DecisionTreeClassifier(random_state=9)))
        models.append(
            ("RF", RandomForestClassifier(n_estimators=num_trees, random_state=9))
        )
        models.append(("GBDT", GradientBoostingClassifier(random_state=9)))
        # models.append(("NB", GaussianNB()))
        # models.append(("SVM", SVC(random_state=9, gamma='auto')))

        for name, model in models:
            kfold = KFold(n_splits=3, random_state=7)
            cv_results = cross_val_score(model, x, y, cv=kfold, scoring="accuracy")
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)  #
            # model.fit(tm.train_features, tm.train_labels)
            # pred = model.predict(tm.valid_features)

            # print("accuracy on test data", accuracy_score(tm.valid_labels, pred))
            # print(confusion_matrix(tm.valid_labels, pred))

        model = GradientBoostingClassifier(random_state=9)
        model.fit(x, y)

        self.clf = model

    def valid_clf(self, df, poolNum=20, splitNum=20):

        df_split = np.array_split(df, splitNum)
        features = []

        with Pool(poolNum) as pool:
            result = list(
                tqdm(pool.imap(parallel_cluster_feat, df_split), total=splitNum)
            )
        # with joblib.Parallel(n_jobs=poolNum, verbose=1) as parallel:
        # result = parallel(joblib.delayed(self.parallel_cluster_feat)(s) for s in df_split)
        for r in result:
            features += r[0]
        features_np = formatND(features)
        labels = np.vstack([r[1] for r in result])  #

        self.bov_helper.cluster(features_np)
        self.bov_helper.developVocabulary(
            n_images=len(features), descriptor_list=features
        )

        self.bov_helper.standardize()
        # self.bov_helper.train(self.bov_helper.mega_histogram, labels)

        pred = self.clf.predict(self.bov_helper.mega_histogram)
        print("accuracy on test data", accuracy_score(labels, pred))
        print(confusion_matrix(labels, pred))

    def add_train_feat(self, df_train, replace=False):
        if (
            os.path.exists("cp_images/train_feat.npy")
            and os.path.exists("cp_images/train_label.npy")
            and (not replace)
        ):
            print("not replace", not replace)
            self.train_features = np.load("cp_images/train_feat.npy")
            self.train_labels = np.load("cp_images/train_label.npy")

        else:
            print("creating feature of training data")
            # self.train_features, self.train_labels = self.parallel_run(df_train)
            self.train_features, self.train_labels = self.generate_features(df_train)
            np.save("cp_images/train_feat.npy", self.train_features)
            np.save("cp_images/train_label.npy", self.train_labels)
            
            
        idx_0 = self.train_labels[:] == 0
        idx_1 = self.train_labels[:] == 1
        
        len_0 = 2000
        
        trn_feat = np.vstack(
            [
                self.train_features[idx_0.ravel()][:len_0, :],
                self.train_features[idx_1.ravel()],
            ]
        )

        trn_label = np.vstack(
            [
                self.train_labels[idx_0.ravel()][:len_0, :],
                self.train_labels[idx_1.ravel()],
            ]
        )

        idx_rnd = np.random.permutation(trn_label.shape[0])

        self.train_features = trn_feat[idx_rnd]
        self.train_labels = trn_label[idx_rnd].ravel()

    def add_test_feat(self, df_test, replace=False):

        if (
            os.path.exists("cp_images/test_feat.npy")
            and os.path.exists("cp_images/test_label.npy")
            and (not replace)
        ):
            self.test_features = np.load("cp_images/test_feat.npy")
            self.test_labels = np.load("cp_images/test_label.npy")
        else:
            self.test_features, self.test_labels = self.generate_features(df_test)
            np.save("cp_images/test_feat.npy", self.test_features)
            np.save("cp_images/test_label.npy", self.test_labels)

    def add_valid_feat(self, df_valid, replace=False):
        if (
            os.path.exists("cp_images/valid_feat.npy")
            and os.path.exists("cp_images/valid_label.npy")
            and (not replace)
        ):
            self.valid_features = np.load("cp_images/valid_feat.npy")
            self.valid_labels = np.load("cp_images/valid_label.npy")
        else:
            self.valid_features, self.valid_labels = self.generate_features(df_valid)

            np.save("cp_images/valid_feat.npy", self.valid_features)
            np.save("cp_images/valid_label.npy", self.valid_labels)

    def add_pred_feat(self, df_pred):

        self.pred_paths = list(df_pred["path"])
        self.pred_features, self.pred_labels = parallel_run(df_pred)

    def cp_fake_img(self, dst_folder, df_pred):
        def add_dst(x):
            name = "_".join(x.split("/")[-2:])
            return os.path.join(dst_folder, name)

        df_copy = df_pred[df_pred.pred == 0]
        df_copy["dst_path"] = df_copy["path"].map(add_dst)
        df_copy[["path", "dst_path"]].values

        return df_copy
    
    def load_model(self, rf_path=None, gbdt_path=None, lr_path=None):
        
        models = []
        
        if rf_path is not None and os.path.exists(rf_path):
            model = pickle.load(open(rf_path, "rb"))
            models.append(model)
    
        if gbdt_path is not None and os.path.exists(gbdt_path):
            model = pickle.load(open(gbdt_path, "rb"))
            models.append(model)
        
        if lr_path is not None and os.path.exists(lr_path):
            model = pickle.load(open(lr_path, "rb"))
            models.append(model)
            
        self.models = models    
    
    def blend_predict(self, img_path, threshold=0.5):
    
        feature = self.extract_features(img_path)
        feature = np.array([feature])
        preds = []
        
        for model in self.models:
            pred = model.predict_proba(feature)[:,1]
            preds.append(pred)
        
        preds = np.hstack(preds)
        
        return (preds.mean() > threshold).astype(np.int16)
            
    def parallel_run(self, df, poolNum=20, splitNum=20):

        df_split = np.array_split(df, splitNum)

        with Pool(poolNum) as pool:
            result = list(
                tqdm(pool.imap(self.generate_features, df_split), total=splitNum)
            )
        # with joblib.Parallel(n_jobs=poolNum, verbose=1) as parallel:
        # result = parallel(joblib.delayed(self.generate_features)(s) for s in df_split)
        features = np.vstack([r[0] for r in result])
        labels = np.vstack([r[1] for r in result])
        # labels = [r[1] for r in result]

        return features, labels


class ImageHelpers:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]

    def features_kaze(self, image, vector_size=32):
        try:
            # Using KAZE, cause SIFT, ORB and other was moved to additional module
            # which is adding addtional pain during install
            alg = cv2.KAZE_create()
            # Dinding image keypoints
            kps = alg.detect(image)
            # Getting first 32 of them.
            # Number of keypoints is varies depend on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            # computing descriptors vector
            kps, dsc = alg.compute(image, kps)
            # Flatten all of them in one big vector - our feature vector
            print(dsc.shape)
            dsc = dsc.flatten()
            # Making descriptor of same size
            # Descriptor vector size is 64
            needed_size = vector_size * 64
            if dsc.size < needed_size:
                # if we have less the 32 descriptors then just adding zeros at the
                # end of our feature vector
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            # print 'Error: ', e
            return None

        return dsc


class BOVHelpers:
    def __init__(self, n_clusters=20):

        self.n_clusters = n_clusters

        self.kmean_path = "log/kmean.sav"
        if os.path.exists(self.kmean_path):
            self.kmeans_obj = pickle.load(open(self.kmean_path, "rb"))
        else:
            self.kmeans_obj = None

        self.scaler_path = "log/scaler.sav"

        if os.path.exists(self.scaler_path):
            self.scale = pickle.load(open(self.scaler_path, "rb"))
        else:
            self.scale = None

        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf = SVC()

    def cluster(self, descriptor_vstack):
        """    
        cluster using KMeans algorithm, 

        """

        if self.kmeans_obj is None:
            self.kmeans_obj = KMeans(n_clusters=self.n_clusters, n_jobs=20)
            self.kmeans_ret = self.kmeans_obj.fit_predict(descriptor_vstack)
            pickle.dump(self.kmeans_obj, open(self.kmean_path, "wb"))
        else:
            self.kmeans_ret = self.kmeans_obj.predict(descriptor_vstack)

        return self.kmeans_ret

    def developVocabulary(self, n_images, descriptor_list, kmeans_ret=None):

        """
        Each cluster denotes a particular visual word 
        Every image can be represeted as a combination of multiple 
        visual words. The best method is to generate a sparse histogram
        that contains the frequency of occurence of each visual word 

        Thus the vocabulary comprises of a set of histograms of encompassing
        all descriptions for all images

        """

        self.mega_histogram = np.array(
            [np.zeros(self.n_clusters) for i in range(n_images)]
        )
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count + j]
                else:
                    idx = kmeans_ret[old_count + j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        # print("Vocabulary Histogram Generated")
        return self.mega_histogram

    def standardize(self, std=None, raw=None):
        """
        
        standardize is required to normalize the distribution
        wrt sample size and features. If not normalized, the classifier may become
        biased due to steep variances.

        """

        if raw is None:
            data = self.mega_histogram
        else:
            data = raw

        if self.scale is None:
            self.scale = StandardScaler().fit(data)
            self.mega_histogram = self.scale.transform(data)
            pickle.dump(self.scale, open(self.scaler_path, "wb"))
        else:
            # print("STD not none. External STD supplied")
            self.mega_histogram = self.scale.transform(data)

        return self.mega_histogram

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions


if __name__ == "__main__":

    df_train, df_test = readIndex()

    df_train = df_train.sample(20000)
    df_test = df_test.sample(3000)

    # print(df_train["label"].value_counts())
    # print(df_test["label"].value_counts())

    fake_paths = ["/data1/1903_face-anti-spoof/valid_spoof3/"]
    true_paths = ["/data/fake_faces_20181123/1/"]

    # pred_paths = glob.glob(
    # "/data1/test_fakeimg_20190315/3/*.jpg", recursive=False
    # ) + glob.glob("/data1/test_fakeimg_20190315/4/*.jpg", recursive=False)

    pred_paths = glob.glob(
        "/data2/Aiaudit_Gmcc/data/754/2019022[2]_JPG/*/[BC].jpg", recursive=False
    )

    df_pred = pd.DataFrame({"path": pred_paths})
    df_pred["label"] = 1

    df_pred = df_pred.sample(100)

    df_valid = readDirect(fake_paths, true_paths)

    df_fake = df_valid[df_valid.label == 0]
    df_real = df_valid[df_valid.label == 1].sample(70)
    df_valid = pd.concat([df_fake, df_pred], axis=0)

    tm = TrainModel()

    # features, features_np, labels =
    # tm.train_cluster(df_train, replace=False)
    # tm.valid_clf(df_valid)

    # sys.exit(0)

    tm.add_train_feat(df_train, False)
    tm.add_test_feat(df_test, False)
    tm.add_valid_feat(df_valid, False)  #
    # tm.add_pred_feat(df_pred)

    # model = GradientBoostingClassifier(random_state=9)
    # model.fit(tm.train_features, tm.train_labels)
    # pred = model.predict(tm.pred_features)

    # df_pred['pred'] = pred

    # df_pred_fake = tm.cp_fake_img('/data2/caiguochen/1903_face_anti_spoof/pred/', df_pred)
    # sys.exit(0)

    models = []
    results = []
    names = []
    scoring = "accuracy"
    models.append(("LR", LogisticRegression(random_state=9, solver="lbfgs")))
    # models.append(("LDA", LinearDiscriminantAnalysis()))
    # models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier(random_state=9)))
    models.append(
        ("RF", RandomForestClassifier(n_estimators=num_trees, random_state=9))
    )
    models.append(("GBDT", GradientBoostingClassifier(random_state=9)))
    # models.append(("NB", GaussianNB()))
    # models.append(("SVM", SVC(random_state=9, gamma='auto')))
    pred_proba = []
    for name, model in models:
        kfold = KFold(n_splits=3, random_state=7)
        cv_results = cross_val_score(
            model,
            tm.train_features,
            tm.train_labels,
            cv=kfold,
            scoring= "accuracy", # "roc_auc",  #
        )
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)  #
        model.fit(tm.train_features, tm.train_labels)
        pred = model.predict(tm.valid_features)
        pickle.dump(model , open('log/%s.sav' %name, "wb"))
        print("accuracy on test data", accuracy_score(tm.valid_labels, pred))
        print(confusion_matrix(tm.valid_labels, pred))
        
        if name in ["GBDT", "RF"]:
            pred_proba.append(model.predict_proba(tm.valid_features)[:,1:])
    
    
    sys.exit(0)
    param_test1 = {
        "n_estimators": range(100, 1001, 100),
        "max_depth": range(5, 16, 2),
        "min_samples_split": range(200, 1001, 200),
        # "min_samples_split": range(1000, 2100, 200),
        # "min_samples_leaf": range(30, 71, 10),
        # "max_features": range(7, 20, 2),
        # "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
    }
    gsearch1 = GridSearchCV(
        estimator=GradientBoostingClassifier(
            learning_rate=0.1,
            # min_samples_split=500,
            min_samples_leaf=50,
            # max_depth=8,
            max_features="sqrt",
            subsample=0.8,
            random_state=1,
        ),
        param_grid=param_test1,
        scoring="accuracy",
        n_jobs=20,
        iid=False,
        cv=5,
        verbose=10
    )
    gsearch1.fit(tm.train_features, tm.train_labels.ravel())
    print(gsearch1.best_params_, gsearch1.best_score_)  #
    pred = gsearch1.best_estimator_.predict(tm.valid_features)

    print("accuracy on test data", accuracy_score(tm.valid_labels, pred))
    print(confusion_matrix(tm.valid_labels, pred))

    pred_proba = gsearch1.best_estimator_.predict_proba(tm.valid_features)

    thresholds = []
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        res = accuracy_score(tm.valid_labels, (pred_proba[:, 1] > thresh).astype(int))
        thresholds.append([thresh, res])
        # print("F1 score at threshold {0} is {1}".format(thresh, res))

    thresholds.sort(key=lambda x: x[1], reverse=True)
    best_thresh = thresholds[0][0]
    best_acc = thresholds[0][1]
    print("Best threshold of capsule: ", best_thresh, best_acc)

    param = {
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "objective": "binary",
        "max_depth": -1,
        "learning_rate": 0.01,
        "min_child_samples": 20,
        "boosting": "gbdt",
        "feature_fraction": 0.9,
        "bagging_freq": 1,
        "bagging_fraction": 0.9,
        "bagging_seed": 11,
        "metric": "auc",
        "lambda_l1": 0.1,
        "verbosity": -1,
        "nthread": 20,
        "random_state": 666,
    }

    trn_data = lgb.Dataset(
        tm.train_features, label=tm.train_labels.ravel()
    )  # , categorical_feature=categorical_feats)
    val_data = lgb.Dataset(
        tm.valid_features, label=tm.valid_labels.ravel()
    )  # , categorical_feature=categorical_feats)

    num_round = 5000
    clf = lgb.train(
        param,
        trn_data,
        num_round,
        valid_sets=[val_data],
        verbose_eval=20,
        early_stopping_rounds=5,
    )
