# -- coding:utf-8 --
from feature_svm import TrainModel
from create_index_face import readIndex, readDirect
from sklearn.metrics import confusion_matrix
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import os, sys, shutil


def do_predict(img_paths):
    tm = TrainModel()
    tm.load_model(rf_path="log/rf.sav", gbdt_path="log/gbdt.sav", lr_path=None)
    preds = []
    for img_path in tqdm(img_paths):
        pred = tm.blend_predict(img_path, 0.4)
        preds.append(pred)
        
    return preds
    
def parallel_predict(df):
    
    split_paths = np.array_split(list(df.path), 20)
    
    with Pool(10) as pool:
        preds = list(
            pool.imap(do_predict, split_paths)
        )
        
    preds = np.hstack(preds)
    
    df['pred'] = preds
    
    return df
    
    
def cp_fake_img(df):

    df_cp = df[(df.label==1 )& (df.pred==0)]
    
    for src_path in list(df_cp.path):
        
        name = os.path.basename(src_path)
        
        dst_path = os.path.join('/data1/test_fakeimg_20190315/pred_fake/', name)
        shutil.copyfile(src_path, dst_path)
        

    

if __name__ == "__main__":

    tqdm.pandas()

    df_train, df_test = readIndex()

    paths = list(df_train.path.head())
    labels = list(df_train.label.head())

    fake_paths = ["/data1/1903_face-anti-spoof/valid_spoof3/"]
    true_paths = ["/data/fake_faces_20181123/1/"]

    # pred_paths = glob.glob(
    # "/data2/Aiaudit_Gmcc/data/754/2019022[2]_JPG/*/[BC].jpg", recursive=False
    # )

    pred_paths = glob.glob(
        "/data1/test_fakeimg_20190315/3/*.jpg", recursive=False
    ) + glob.glob("/data1/test_fakeimg_20190315/4/*.jpg", recursive=False)

    df_pred = pd.DataFrame({"path": pred_paths})
    df_pred["label"] = 1

    df_pred = df_pred.sample(20000)

    df_valid = readDirect(fake_paths, true_paths)

    df_fake = df_valid[df_valid.label == 0]
    df_real = df_valid[df_valid.label == 1].sample(70)
    df_valid = pd.concat([df_fake, df_pred], axis=0)

    

    df_valid = parallel_predict(df_valid)

    print(confusion_matrix(df_valid.label, df_valid.pred))
    
    cp_fake_img(df_valid)

