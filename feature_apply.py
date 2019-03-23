# -- coding:utf-8 --

from feature_train import AntiModel
import numpy as np
from tqdm import tqdm
import glob
    

if __name__ == "__main__":

    paths = glob.glob('valid/valid_0/*.jpg')
    anti = AntiModel()
    
    preds = []
    for path in tqdm(paths):
        pred = anti.predict_path(path)
        preds.append(pred)

