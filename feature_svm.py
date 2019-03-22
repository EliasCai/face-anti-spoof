# -- coding:utf-8 --

import os, glob, sys
from tqdm import tqdm

from create_index_face import readIndex, readDirect
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from sklearn.externals import joblib

# import lightgbm as lgb
import numpy as np
import pandas as pd
import mahotas
import cv2
from skimage import feature
import pickle


# fixed-sizes for image
fixed_size = tuple((500, 500))

# no.of.trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# seed for reproducing same results
seed = 9


class AntiModel:
    def __init__(self):
        extract_feature = ExtractFeature()


def do_sift_batch(image_paths):

    features = []
    sift_object = cv2.xfeatures2d.SIFT_create()
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        image = cv2.resize(image, fixed_size)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = sift_object.detectAndCompute(gray, None)
        features.append(des)

    features = np.vstack(features)
    # print('do_sift', features.shape)

    return features


def do_sift_single(image_path):

    features = []
    sift_object = cv2.xfeatures2d.SIFT_create()

    image = cv2.imread(image_path)
    image = cv2.resize(image, fixed_size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = sift_object.detectAndCompute(gray, None)

    # print('do_sift', des.shape())

    return des


def parallel_sift(image_paths, poolNum=20, single=False):

    if single:
        with joblib.Parallel(n_jobs=poolNum, verbose=1) as parallel:
            features = parallel(joblib.delayed(do_sift_single)(s) for s in image_paths)
    else:
        split_paths = np.array_split(image_paths, poolNum)
        with joblib.Parallel(n_jobs=poolNum, verbose=1) as parallel:
            features = parallel(joblib.delayed(do_sift_batch)(s) for s in split_paths)

        features = np.vstack(features)

    return features


def transform_vocabulary(feature):

    left = pd.Series(np.zeros(128), index=range(128))
    s = pd.Series(feature)
    right = s.value_counts()
    a = left.align(right, join="left", fill_value=0)
    return a[1].values


def prepare_cluster(
    images_paths, kmean_path="log/kmean.sav", replace=False, n_clusters=100
):

    if os.path.exists(kmean_path) and (not replace):
        pass
        # self.kmeans_obj = pickle.load(open(self.kmean_path, "rb"))
    else:
        feature = parallel_sift(images_paths)
        print(feature.shape)
        kmeans_obj = KMeans(n_clusters=n_clusters, n_jobs=20)
        kmeans_obj.fit(feature)
        pickle.dump(kmeans_obj, open(kmean_path, "wb"))


def prepare_scaler(
    images_paths,
    kmean_path="log/kmean.sav",
    scaler_path="log/scaler.sav",
    replace=False,
):

    if os.path.exists(scaler_path) and (not replace):
        pass
        # self.scale = pickle.load(open(self.scaler_path, "rb"))
    else:
        kmeans_obj = pickle.load(open(kmean_path, "rb"))
        feature = parallel_sift(images_paths, single=True)
        feature = list(map(kmeans_obj.predict, feature))
        feature = list(map(transform_vocabulary, feature))
        feature = np.vstack(feature)
        scaler = StandardScaler()
        feature = scaler.fit_transform(feature)
        pickle.dump(scaler, open(scaler_path, "wb"))

    # return feature


class ExtractFeature:
    def __init__(self, kmean_path="log/kmean.sav", scaler_path="log/scaler.sav"):

        self.sift_object = cv2.xfeatures2d.SIFT_create()
        self.kmeans_obj = pickle.load(open(kmean_path, "rb"))
        self.scaler = pickle.load(open(scaler_path, "rb"))

    # feature-descriptor-4: LBP
    def fd_lbp(self, image, numPoints=24, radius=8, eps=1e-7):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(
            lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2)
        )

        # normalize the histogram
        hist = hist.astype("float")
        hist /= hist.sum() + eps
        return hist

    # feature-descriptor-3: Color Histogram
    def fd_histogram(self, image, mask=None):
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

    # feature-descriptor-2: Haralick Texture
    def fd_haralick(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        return haralick

    # feature-descriptor-1: Hu Moments
    def fd_hu_moments(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def fd_sift(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift_object.detectAndCompute(gray, None)
        feature = np.array(des)
        kmeans_ret = self.fd_cluster(feature)
        vocabulary = self.transform_vocabulary(kmeans_ret)
        standard = self.scaler.transform(vocabulary.reshape(1, -1))
        return standard.ravel()

    def transform_vocabulary(self, feature):

        left = pd.Series(np.zeros(128), index=range(128))
        s = pd.Series(feature)
        right = s.value_counts()
        a = left.align(right, join="left", fill_value=0)
        return a[1].values

    def fd_cluster(self, feature):

        kmeans_ret = self.kmeans_obj.predict(feature)

        return kmeans_ret

    def extract_features(self, img_path):

        image = cv2.imread(img_path)
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = self.fd_hu_moments(image)
        fv_haralick = self.fd_haralick(image)
        fv_histogram = self.fd_histogram(image)
        fv_lbp = self.fd_lbp(image)
        fv_sift = self.fd_sift(image)
        feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments, fv_lbp, fv_sift])
        return feature

    def generate_features(self, df):

        features, labels = [], []

        for rowid, row in tqdm(df.iterrows()):
            # for rowid, row in df.iterrows():

            img_path = row["path"]
            feature = self.extract_features(img_path)
            labels.append(row["label"])
            features.append(feature)

        features = np.vstack(features)
        labels = np.array(labels).reshape((-1, 1))
        return features, labels


def parallel_features(df):
    et = ExtractFeature()
    features, labels = et.generate_features(df)
    return features, labels


def main_data(df, poolNum=20, replace=False):

    split_df = np.array_split(df, poolNum)

    with joblib.Parallel(n_jobs=poolNum, verbose=0) as parallel:
        # result = parallel(joblib.delayed(parallel_features)(d) for d in split_df)
        result = parallel(joblib.delayed(parallel_features)(d) for d in split_df)

    features = np.vstack([r[0] for r in result])
    labels = np.vstack([r[1] for r in result])

    return features, labels


def train_data(df, replace=False):

    feat_path = "data/train_feat.npy"
    label_path = "data/train_label.npy"

    if os.path.exists(feat_path) and os.path.exists(label_path) and (not replace):

        features = np.load(feat_path)
        labels = np.load(label_path)

    else:
        print("creating feature of training data")
        features, labels = main_data(df, replace=replace)
        np.save(feat_path, features)
        np.save(label_path, labels)

    return features, labels.ravel()


def test_data(df, replace=False):

    feat_path = "data/test_feat.npy"
    label_path = "data/test_label.npy"

    if os.path.exists(feat_path) and os.path.exists(label_path) and (not replace):

        features = np.load(feat_path)
        labels = np.load(label_path)

    else:
        print("creating feature of training data")
        features, labels = main_data(df, replace=replace)
        np.save(feat_path, features)
        np.save(label_path, labels)

    return features, labels.ravel()

    
def valid_data(replace=False):
    
    fake_paths = glob.glob('valid/valid_0/*.jpg')
    true_paths = glob.glob('valid/valid_1/*.jpg')
    
    df_fake = pd.DataFrame({'path': fake_paths})
    df_fake['label'] = 0
    df_true = pd.DataFrame({'path': true_paths})
    df_true['label'] = 1
    
    df = pd.concat([df_true, df_fake], axis=0).sample(frac=1)
    
    feat_path = "data/valid_feat.npy"
    label_path = "data/valid_label.npy"

    if os.path.exists(feat_path) and os.path.exists(label_path) and (not replace):

        features = np.load(feat_path)
        labels = np.load(label_path)

    else:
        print("creating feature of training data")
        features, labels = main_data(df, replace=replace)
        np.save(feat_path, features)
        np.save(label_path, labels)

    return features, labels.ravel()

def main():

    path = "/data1/test_fakeidcard_img/train/20190308163511733747_8.jpg"
    df_train, df_test = readIndex()
    image = cv2.imread(path)
    images = [image] * 20
    paths = [path] * 20
    df_train, df_test = readIndex()
    et = ExtractFeature()

    prepare_cluster(list(df_train.sample(1000).path), replace=False)
    prepare_scaler(list(df_train.sample(1000).path), replace=False)

    # features, labels = et.generate_features(df_train.sample(10))

    split_df = np.array_split(df_train.sample(200), 20)
    with joblib.Parallel(n_jobs=20, verbose=1) as parallel:
        # result = parallel(joblib.delayed(parallel_features)(d) for d in split_df)
        result = parallel(joblib.delayed(parallel_features)(d) for d in split_df)

    features = np.vstack([r[0] for r in result])
    labels = np.vstack([r[1] for r in result])

    return features, labels


if __name__ == "__main__":

    df_train, df_test = readIndex()

    # features, labels = train_data(df_train.sample(20000), replace=False)
    features, labels = valid_data(replace=True)
    print(features.shape, labels.shape)
