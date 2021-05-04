import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import cv2

from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split

import albumentations as A
import torchvision.transforms as T

# import shutil
# shutil.rmtree('./aug_re_img')

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
                                                      random_state=1,
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

print(class_labels)
train_csv.label.value_counts()

pre_dict = {}
for _, row in train_csv.iterrows():
    if row.label not in pre_dict:
        pre_dict[row.label] = [row.image]
    else:
        pre_dict[row.label] += [row.image]


def img_read(strimg):
    root = '../input/plant-pathology-2021-fgvc8/train_images'
    imgpath = os.path.join(root, strimg)
    img = Image.open(imgpath)

    return img


img = img_read('802f7439ec1ef0cd.jpg')
plt.imshow(np.array(img))

# fig, axs = plt.subplots(2,5,figsize=(30,20))
# c=0
# for i in range(2):
#     for j in range(5):

#         axs[i][j].imshow(np.array(zz[c]))
#         c+=1

os.mkdir('aug_re_img')
ROOT='./aug_re_img'
train_new =[]

for i in range(12):
    print(f'Start to execute {i} class')
    print('--------------------------------')
    if i in [3,9]:
        for img_path in tqdm(pre_dict[i]):
            if np.random.randint(0,5)<4:
                img = img_read(img_path)
                imgor = T.Resize((256,256))(img)
                newpath = os.path.join(ROOT,img_path)
                cv2.imwrite(newpath,np.array(imgor))
                train_new.append([img_path,i])
    if i in [1]:
        for img_path in tqdm(pre_dict[i]):
            img = img_read(img_path)
            imgor = T.Resize((256,256))(img)
            newpath = os.path.join(ROOT,img_path)
            cv2.imwrite(newpath,np.array(imgor))
            train_new.append([img_path,i])
    if i in [0,6]:
        ptransform =A.Compose([
            A.ShiftScaleRotate(p=1),
                A.Resize(256,256)])
        for img_path in tqdm(pre_dict[i]):
            img = img_read(img_path)
            imgor = T.Resize((256,256))(img)
            imgtr =ptransform(image=np.array(img))['image']
            newpath = os.path.join(ROOT,img_path)
            newpath1 = os.path.join(ROOT,'1$'+img_path)
            cv2.imwrite(newpath,np.array(imgor))
            cv2.imwrite(newpath1,imgtr)
            train_new.append([img_path,i])
            train_new.append(['1$'+img_path,i])
    if i in [4]:
        ptransform =A.Compose([
            A.ShiftScaleRotate(p=1),
                A.Resize(256,256)])
        for img_path in tqdm(pre_dict[i]):
            img = img_read(img_path)
            imgor = T.Resize((256,256))(img)
            newpath = os.path.join(ROOT,img_path)
            cv2.imwrite(newpath,np.array(imgor))
            train_new.append([img_path,i])
            for I in range(2):
                imgtr =ptransform(image=np.array(img))['image']
                newpath1 = os.path.join(ROOT,f'{I}$'+img_path)
                cv2.imwrite(newpath1,imgtr)
                train_new.append([f'{I}$'+img_path,i])
    if i in [10]:
        ptransform =A.Compose([
            A.ShiftScaleRotate(p=1),
            A.Flip(p=1),
                A.Resize(256,256)])
        for img_path in tqdm(pre_dict[i]):
            img = img_read(img_path)
            imgor = T.Resize((256,256))(img)
            newpath = os.path.join(ROOT,img_path)
            cv2.imwrite(newpath,np.array(imgor))
            train_new.append([img_path,i])
            for I in range(4):
                imgtr =ptransform(image=np.array(img))['image']
                newpath1 = os.path.join(ROOT,f'{I}$'+img_path)
                cv2.imwrite(newpath1,imgtr)
                train_new.append([f'{I}$'+img_path,i])
    if i in [2,11]:
        ptransform =A.Compose([
            A.ShiftScaleRotate(p=1),
            A.Flip(p=1),
            A.Rotate(p=1),
                A.Resize(256,256)])
        for img_path in tqdm(pre_dict[i]):
            img = img_read(img_path)
            imgor = T.Resize((256,256))(img)
            newpath = os.path.join(ROOT,img_path)
            cv2.imwrite(newpath,np.array(imgor))
            train_new.append([img_path,i])
            for I in range(15):
                imgtr =ptransform(image=np.array(img))['image']
                newpath1 = os.path.join(ROOT,f'{I}$'+img_path)
                cv2.imwrite(newpath1,imgtr)
                train_new.append([f'{I}$'+img_path,i])
    if i in [5,7,8]:
        ptransform =A.Compose([
            A.ShiftScaleRotate(p=1),
            A.Flip(p=1),
            A.Rotate(p=1),
                A.Resize(256,256)])
        for img_path in tqdm(pre_dict[i]):
            img = img_read(img_path)
            imgor = T.Resize((256,256))(img)
            newpath = os.path.join(ROOT,img_path)
            cv2.imwrite(newpath,np.array(imgor))
            train_new.append([img_path,i])
            for I in range(28):
                imgtr =ptransform(image=np.array(img))['image']
                newpath1 = os.path.join(ROOT,f'{I}$'+img_path)
                cv2.imwrite(newpath1,imgtr)
                train_new.append([f'{I}$'+img_path,i])


dftrain = pd.DataFrame(train_new,columns=['image','label'])
dftrain.to_csv('train.csv',index=False)
val_csv.to_csv('val.csv',index=False)
