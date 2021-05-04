import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import cv2

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import albumentations as A

df=pd.read_csv('/kaggle/input/plant-pathology-2021-fgvc8/train.csv')

df_d = pd.read_csv('../input/jyduplicate/duplicates.csv',header = None)
df_d.columns = ['c1','c2']
def conbine_du(df_ta,df_d):
    for x,y in df_d.values:
        l1,l2 = df_ta[df_ta.image ==x].values[0][1].split(' '),df_ta[df_ta.image ==y].values[0][1].split(' ')
        labels_co = list(set(l1+l2))
        delimeter = ' '
        df_ta.loc[df_ta[df_ta.image ==x].index,'labels'] = delimeter.join(labels_co)
        df_ta = df_ta.drop(df_ta[df_ta.image ==y].index)
    return df_ta
df_redu = conbine_du(df,df_d)

def ll(x):
    if x =='frog_eye_leaf_spot rust':
        return 'rust frog_eye_leaf_spot'
    if x =='complex rust':
        return 'rust complex'
    else:
        return x
df_redu.labels = df_redu.labels.apply(lambda x: ll(x))


class_labels = df_redu.labels.value_counts().index.tolist()
def label_exchange(df):
    le = LabelEncoder()
    le.fit(class_labels)
    df['label_ex'] = le.transform(df.labels.values)
    return df, le.classes_

df,class_labels = label_exchange(df_redu)


def data_split(phase='train', size=0.2):
    x_train, x_val, y_train, y_val = train_test_split(df.image, df.label_ex,
                                                      random_state=42,
                                                      shuffle=True,
                                                      test_size=size,
                                                      stratify=df.label_ex)
    tar_csv = pd.DataFrame()
    if phase in ['train']:
        tar_csv['image'] = x_train
        tar_csv['label'] = y_train
    elif phase in ['val']:
        tar_csv['image'] = x_val
        tar_csv['label'] = y_val
    elif phase in ['test']:
        DIR = '../input/plant-pathology-2021-fgvc8/sample_submission.csv'
        tar_csv = pd.read_csv(DIR)

    return tar_csv


TEST_SIZE = 0.2
train_csv = data_split(phase='train', size=TEST_SIZE)
val_csv = data_split(phase='val', size=TEST_SIZE)
print(f'The test size is {TEST_SIZE}\nThe length of train set is {len(train_csv)}')
print(f'The length of validation set is {len(val_csv)}')

def img_read(strimg):
    root = '../input/plant-pathology-2021-fgvc8/train_images'
    imgpath =os.path.join(root,strimg)
    img = cv2.imread(imgpath,cv2.COLOR_BGR2RGB)
    return img

for img in tqdm(val_csv.image.tolist()):
    imageo = img_read(img)
    imgor =  A.Resize(256,256)(image = imageo)['image']
    cv2.imwrite(img,imgor)
