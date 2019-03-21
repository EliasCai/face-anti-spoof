# -- coding:utf-8 --
import glob, sys, os, codecs, random, shutil
import pandas as pd
from itertools import groupby
import numpy as np
import re
import cv2
from os.path import join
from sklearn.model_selection import train_test_split
from tqdm import tqdm


print("create_index version = 2.0")


def createIndex(fake_paths, true_paths):  # 创建训练集和测试集的索引文件

    split_label = lambda x: x.split("/")[-2]
    # find_label = lambda x: True if re.match('^[0-9]_', x) else False
    # split_label =  lambda x: list(filter(find_label, x.split('/')))[0].split('_')[0]
    train_csv = "train_face.csv"
    test_csv = "test_face.csv"

    if os.path.exists(train_csv) & os.path.exists(test_csv) & False:
        print("file exists!")

    else:
        print("file not exists!")
        fake_img_paths = []
        label_list = []
        for path in fake_paths:
            search_path = os.path.join(path, "**/*.png")
            if len(search_path) == 0:
                continue
            search_res = glob.glob(search_path, recursive=True)
            print(path, len(search_res))
            random.shuffle(search_res)
            print("total len of ", len(search_res))
            search_res = search_res[:7000]
            fake_img_paths += search_res

        label_list += [0] * len(fake_img_paths)
        # print(fake_img_paths[:5])
        print("total num of fake img =", len(fake_img_paths))

        true_img_paths = []
        for path in true_paths:
            search_path = os.path.join(path, "**/*.png")

            if len(search_path) == 0:
                continue
            search_res = glob.glob(search_path, recursive=True)
            # search_res = [s for s in search_res if re.search('0002.png',s)] # 只选取现场照片
            print(path, len(search_res))
            random.shuffle(search_res)
            print("total len of ", len(search_res))
            search_res = search_res[:28000]
            true_img_paths += search_res
            # true_img_paths += glob.glob(search_path, recursive=True)

        label_list += [1] * len(true_img_paths)
        # print(true_img_paths[:5])
        print("total num of img =", len(true_img_paths))
        data = pd.DataFrame(
            {"path": fake_img_paths + true_img_paths, "label": label_list}
        )

        idx_fake = data["label"] == 0

        # data = pd.concat([data.loc[idx_fake,['path', 'label']].sample(10000),
        # data.loc[~idx_fake,['path', 'label']].sample(10000)],axis=0).sample(frac=1)

        train_len = int(data.shape[0] * 0.8)
        # data.iloc[:train_len].to_csv(train_csv, index=False, header=None)
        # data.iloc[train_len:].to_csv(test_csv, index=False, header=None)
        # return data
        # img_paths = sorted(fake_img_paths + true_img_paths, key=label_list)
        # print(img_paths[:5])
        # sys.exit(0)
        # return fake_img_paths + true_img_paths,label_list
        img_label = [
            (a, b) for a, b in zip(fake_img_paths + true_img_paths, label_list)
        ]
        img_groups = groupby(img_label, lambda x: x[1])
        with codecs.open(train_csv, "w") as f_train:
            with codecs.open(test_csv, "w") as f_test:
                for label, group in img_groups:

                    print(label, group)

                    files = list(group)
                    files = [f[0] for f in files]
                    np.random.shuffle(files)
                    # if label == 0:
                    # files = files[:700]
                    train_len = min(32000, int(len(files) * 0.8))
                    both_len = min(40000, len(files))
                    # print(label, len(files), train_len, both_len)
                    for file in files[:train_len]:
                        f_train.write(
                            file + "," + str(label) + "\n"
                        )  # ','.join(label.split('_'))

                    for file in files[train_len:both_len]:
                        f_test.write(file + "," + str(label) + "\n")


def readDirect(fake_paths, true_paths):  # 创建训练集和测试集的索引文件

    split_label = lambda x: x.split("/")[-2]

    fake_img_paths = []
    label_list = []
    for path in fake_paths:
        search_path = os.path.join(path, "**/*.jpg")
        if len(search_path) == 0:
            continue
        search_res = glob.glob(search_path, recursive=True)
        print(path, len(search_res))
        random.shuffle(search_res)
        search_res = search_res[:7000]
        fake_img_paths += search_res

    label_list += [0] * len(fake_img_paths)
    print("total num of fake img =", len(fake_img_paths))

    true_img_paths = []
    for path in true_paths:
        search_path = os.path.join(path, "**/*.jpg")

        if len(search_path) == 0:
            continue
        search_res = glob.glob(search_path, recursive=True)
        # search_res = [s for s in search_res if re.search('0002.png',s)]
        print(path, len(search_res))
        random.shuffle(search_res)
        search_res = search_res[:7000]
        true_img_paths += search_res
        # true_img_paths += glob.glob(search_path, recursive=True)

    label_list += [1] * len(true_img_paths)
    # print(true_img_paths[:5])
    print("total num of true img =", len(true_img_paths))
    data = pd.DataFrame({"path": fake_img_paths + true_img_paths, "label": label_list})

    return data


def readIndex():  # 读取训练集和测试集的索引文件

    train_csv = os.path.join("train.csv")
    test_csv = os.path.join("test.csv")
    if os.path.exists(train_csv) & os.path.exists(test_csv):
        print("file exists!")
        df_train = pd.read_csv(
            train_csv,
            # names=["path", "label"],  #'label_name'],
            # dtype={"label": np.int32},
        )

        df_test = pd.read_csv(
            test_csv,
            # names=["path", "label"],  # 'label_name'],
            # dtype={"label": np.int32},
        )

        return df_train[df_train["label"] != 30], df_test[df_test["label"] != 30]
    else:
        print("file not exists!")
        sys.exit(0)


def read_extern(path="extern_data/client_train_raw.txt", label=1):
    # label: 0-fake; 1-true
    df = pd.read_csv(path, header=0)
    df.columns = ["path"]
    if label == 1:
        df["path"] = df["path"].map(
            lambda x: os.path.join("extern_data/ClientRaw", x.replace("\\", "/"))
        )
    else:
        df["path"] = df["path"].map(
            lambda x: os.path.join("extern_data/ImposterRaw", x.replace("\\", "/"))
        )
    df["label"] = label

    return df


def createIndex2():

    train_csv = "train.csv"
    test_csv = "test.csv"

    if os.path.exists(train_csv) & os.path.exists(test_csv) & False:
        return

    fake = glob.glob(
        join("/data2/ai_langchao/Remakeface_PC/remake/logo/", "*.jpg")
    ) + glob.glob(join("/data2/ai_langchao/Remakeface_PC/remake/nologo/", "*.jpg"))

    fake += (
        glob.glob(join("/data2/ai_langchao/Remake_PHONE2/remake_PHONE/", "*.jpg"))
        + glob.glob(join("/data/fake_faces_label_1126/0", "*.jpg"))
        + glob.glob(join("/data2/ai_pingyunbz/1126-1127/DG_6", "*.jpg"))
    )
    
    # fake += (
        # glob.glob(
            # join("/data/img_class_label_DG_20181117/certificate_class/[34]/", "*.jpg")
        # )
        # + glob.glob(
            # join("/data/img_class_label_FS_20181119/certificate_class/[34]/", "*.jpg")
        # )
        # + glob.glob(
            # join("/data/img_class_label_GZ_20181116/certificate_class/[34]/", "*.jpg")
        # )
        # + glob.glob(
            # join("/data/img_class_label_second_20181123/certificate_class/[34]/", "*.jpg")
        # )
        # + glob.glob(
            # join("/data/img_class_label_SZ_20181119/certificate_class/[34]/", "*.jpg")
        # )
        # + glob.glob(
        # "/data2/Aiaudit_Gmcc/data/754/2019030*_JPG/*/[BC].jpg", recursive=True)
    # )
    
    real = []
    real = glob.glob(join("/data2/ai_langchao/Remakeface_PC/source/", "*.jpg"))
    real += glob.glob(join("/data2/ai_langchao/Remake_PHONE2/source/", "*.jpg")) 
    # real += (
        # glob.glob(
            # join("/data/img_class_label_DG_20181117/certificate_class/[346]/", "*.jpg")
        # )
        # + glob.glob(
            # join("/data/img_class_label_FS_20181119/certificate_class/[346]/", "*.jpg")
        # )
        # + glob.glob(
            # join("/data/img_class_label_GZ_20181116/certificate_class/[346]/", "*.jpg")
        # )
        # + glob.glob(
            # join("/data/img_class_label_second_20181123/certificate_class/[346]/", "*.jpg")
        # )
        # + glob.glob(
            # join("/data/img_class_label_SZ_20181119/certificate_class/[346]/", "*.jpg")
        # )
    # )

    real += (
        glob.glob("/data2/Aiaudit_Gmcc/data/660/**/[BC].jpg", recursive=False)
        + glob.glob("/data2/Aiaudit_Gmcc/data/668/**/[BC].jpg", recursive=False) # 
        + glob.glob("/data2/Aiaudit_Gmcc/data/662/**/[BC].jpg", recursive=False)
        + glob.glob(
        "/data2/Aiaudit_Gmcc/data/754/2019030*_JPG/*/[ABC].jpg", recursive=False)
    )

    df_fake = pd.DataFrame({"path": fake})
    df_fake["label"] = 0

    df_real = pd.DataFrame({"path": real})
    df_real["label"] = 1

    shape_0 = min(df_fake.shape[0], df_real.shape[0])

    df_fake = df_fake.sample(shape_0)
    df_real = df_real.sample(shape_0)

    df = pd.concat([df_fake, df_real], axis=0)
    
    print('all image =', df.shape)
    
    # df = df.sample(1000)
    
    df_train, df_test = train_test_split(df, test_size=0.3)
    cp_train = lambda x: os.path.join( 'cp_images/train', os.path.basename(x))
    cp_test = lambda x: os.path.join( 'cp_images/test', os.path.basename(x))
    
    shutil.rmtree('cp_images/train')
    shutil.rmtree('cp_images/test')
    os.mkdir('cp_images/train')
    os.mkdir('cp_images/test')
    
    
    new_train_path = df_train.path.map(cp_train)
    new_test_path = df_test.path.map(cp_test)
    
    def cp_images(scr_paths, dst_paths):
        for s, d in tqdm(zip(scr_paths, dst_paths)):
            # shutil.copyfile(s, d)
            if os.path.exists(d):
                continue
            img = cv2.imread(s)
            img = cv2.resize(img, (300,300))
            cv2.imwrite(d, img)
    
    cp_images(list(df_train['path']), list(new_train_path))
    cp_images(list(df_test['path']), list(new_test_path))
    
    df_train['path'] = new_train_path
    df_test['path'] = new_test_path
    
    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)


if __name__ == "__main__":

    # paths = ['/data/img_class_Install_201810_Finsh/',
    # '/data/img_class_check_train_20180903/',
    # '/data1/1808_img-class/data/']

    fake_paths = [
        "/data2/Aiaudit_Gmcc/train/Model_Fakeface/20190128/DG_6/",
        "/data2/Aiaudit_Gmcc/train/Model_Fakeface/20190128/GZ_6/",
        "/data2/Aiaudit_Gmcc/train/Model_Fakeface/20190128/second_6/",
        "/data2/Aiaudit_Gmcc/train/Model_Fakeface/20190128/SZ_6/",
    ]
    true_paths = ["/data2/Aiaudit_Gmcc/train/Model_FaceCompare/test_160_20190124/"]

    # createIndex(fake_paths, true_paths)
    createIndex2()
    df_train, df_test = readIndex()
    print(df_train["label"].value_counts())
    print(df_test["label"].value_counts())
