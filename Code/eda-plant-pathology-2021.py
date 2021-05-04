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

df=pd.read_csv('/plant-pathology-2021-fgvc8/train.csv')

df_d = pd.read_csv('duplicates.csv',header = None)
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

#df_redu.to_csv('train_redu.csv',index = False)

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

print(class_labels)
train_csv.label.value_counts()


def img_read(strimg, clach=False):
    root = '../input/plant-pathology-2021-fgvc8/train_images'
    imgpath = os.path.join(root, strimg)
    img = cv2.imread(imgpath, cv2.COLOR_BGR2RGB)

    if clach:
        img = A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1)(image=img)['image']
    return img


pre_dict = {}
for _, row in train_csv.iterrows():
    if row.label not in pre_dict:
        pre_dict[row.label] = [row.image]
    else:
        pre_dict[row.label] += [row.image]


fig, axs = plt.subplots(3,4,figsize=(30,20))
cc =0
for i in range(3):
    for j in range(4):
        axs[i][j].axis('off')
        axs[i][j].imshow(img_read(pre_dict[cc][1]))
        axs[i][j].set_title(class_labels[cc])
        cc+=1


fig, axs = plt.subplots(3,4,figsize=(30,20))
cc =0
for i in range(3):
    for j in range(4):
        axs[i][j].axis('off')
        axs[i][j].imshow(img_read(pre_dict[cc][1],clach =True))
        axs[i][j].set_title(class_labels[cc])
        cc+=1


# https://blog.csdn.net/qq_34107425/article/details/107503132 clahe
def imgread(strimg, transform=False):
    root = '../input/plant-pathology-2021-fgvc8/train_images'
    imgpath = os.path.join(root, strimg)
    img = cv2.imread(imgpath, cv2.COLOR_BGR2RGB)

    trlst = [A.HorizontalFlip(p=1),
             A.VerticalFlip(p=1),
             A.Rotate(p=1),
             A.Blur(blur_limit=50, p=1),
             A.ColorJitter(p=1),
             A.ColorJitter(p=1),
             # A.ShiftScaleRotate(p=1),
             A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
             A.FancyPCA(alpha=1, p=1),
             # A.Rotate(p=1),
             # A.RandomSunFlare(p=1),
             A.RandomFog(p=1),
             A.RandomBrightness(p=1),
             #              A.RGBShift(p=1),
             A.RandomSnow(p=1),
             A.RandomContrast(limit=0.5, p=1),
             A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]
    if transform:
        for trans in trlst:
            img2 = trans(image=img)['image']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))
            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(img)
            ax2.imshow(img2)
    else:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    return img


zz = imgread(pre_dict[5][3], transform=True)
zz.shape


# import shutil
# shutil.rmtree('./aug_re_img')
os.mkdir('aug_re_img')
ROOT='./aug_re_img'
train_new =[]

for i in range(12):
    print(f'Start to execute {i} class')
    print('--------------------------------')
    if i in [3, 9]:  #
        for img_path in tqdm(pre_dict[i]):
            transform = A.Compose([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
                A.HorizontalFlip(0.5),
                A.VerticalFlip(0.5),
                A.Rotate(p=0.5),
                A.Resize(256, 256)])

            img = img_read(img_path)
            imgor = transform(image=img)['image']
            newpath = os.path.join(ROOT, img_path)
            cv2.imwrite(newpath, imgor)
            train_new.append([img_path, i])
    if i in [1]:  #
        for img_path in tqdm(pre_dict[i]):

            transform = A.Compose(
                [A.OneOf([
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                    A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
                    A.FancyPCA(alpha=1, p=1)], p=1),
                    A.OneOf([A.HorizontalFlip(p=1),
                             A.VerticalFlip(p=1),
                             A.Rotate(p=1), ], p=1),
                    A.Resize(256, 256)])

            img = img_read(img_path)
            imgor = A.Resize(256, 256)(image=img)['image']
            newpath = os.path.join(ROOT, img_path)
            cv2.imwrite(newpath, imgor)
            train_new.append([img_path, i])

            if np.random.randint(1, 9) > 4:
                imgtr = transform(image=img)['image']
                newpath = os.path.join(ROOT, '1$' + img_path)
                cv2.imwrite(newpath, imgtr)
                train_new.append(['1$' + img_path, i])

    if i in [0, 6]:  #
        for I in range(4):
            for img_path in tqdm(pre_dict[i]):
                transform = A.Compose(
                    [A.OneOf([
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                        A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
                        A.FancyPCA(alpha=1, p=1)], p=1),
                        A.OneOf([A.HorizontalFlip(p=1),
                                 A.VerticalFlip(p=1),
                                 A.Rotate(p=1), ], p=1),
                        A.ShiftScaleRotate(p=1),
                        A.Resize(256, 256)])

                img = img_read(img_path)
                if I == 0:
                    imgor = A.Resize(256, 256)(image=img)['image']
                    newpath = os.path.join(ROOT, img_path)
                    cv2.imwrite(newpath, imgor)
                    train_new.append([img_path, i])
                else:
                    if np.random.randint(0, 9) > 3:
                        imgtr = transform(image=img)['image']
                        newpath = os.path.join(ROOT, f'{I}$' + img_path)
                        cv2.imwrite(newpath, imgtr)
                        train_new.append([f'{I}$' + img_path, i])

    if i in [4]:
        for I in range(4):
            for img_path in tqdm(pre_dict[i]):
                transform = A.Compose(
                    [A.OneOf([
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                        A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
                        A.FancyPCA(alpha=1, p=1)], p=1),
                        A.OneOf([A.HorizontalFlip(p=1),
                                 A.VerticalFlip(p=1),
                                 A.Rotate(p=1), ], p=1),
                        A.ShiftScaleRotate(p=1),
                        A.Resize(256, 256)])

                img = img_read(img_path)

                if I == 0:
                    imgor = A.Resize(256, 256)(image=img)['image']
                    newpath = os.path.join(ROOT, img_path)
                    cv2.imwrite(newpath, imgor)
                    train_new.append([img_path, i])
                else:
                    imgtr = transform(image=img)['image']
                    newpath = os.path.join(ROOT, f'{I}$' + img_path)
                    cv2.imwrite(newpath, imgtr)
                    train_new.append([f'{I}$' + img_path, i])

    if i in [10]:
        for I in range(7):
            for img_path in tqdm(pre_dict[i]):
                transform = A.Compose(
                    [A.OneOf([
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                        A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
                        A.RGBShift(p=1),
                        A.FancyPCA(alpha=1, p=1)], p=1),
                        A.OneOf([A.HorizontalFlip(p=1),
                                 A.VerticalFlip(p=1),
                                 A.Rotate(p=1), ], p=1),
                        A.ShiftScaleRotate(p=1),
                        A.Resize(256, 256)])

                img = img_read(img_path)

                if I == 0:
                    imgor = A.Resize(256, 256)(image=img)['image']
                    newpath = os.path.join(ROOT, img_path)
                    cv2.imwrite(newpath, imgor)
                    train_new.append([img_path, i])
                else:
                    imgtr = transform(image=img)['image']
                    newpath = os.path.join(ROOT, f'{I}$' + img_path)
                    cv2.imwrite(newpath, imgtr)
                    train_new.append([f'{I}$' + img_path, i])

    if i in [2, 11]:
        for I in range(40):
            for img_path in tqdm(pre_dict[i]):
                transform = A.Compose(
                    [A.OneOf([
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                        A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
                        A.RGBShift(p=1),
                        A.ColorJitter(p=1),
                        A.FancyPCA(alpha=1, p=1)], p=1),
                        A.OneOf([A.HorizontalFlip(p=1),
                                 A.VerticalFlip(p=1),
                                 A.Rotate(p=1), ], p=1),
                        A.ShiftScaleRotate(p=1),
                        A.Resize(256, 256)])

                img = img_read(img_path)
                if I == 0:
                    imgor = A.Resize(256, 256)(image=img)['image']
                    newpath = os.path.join(ROOT, img_path)
                    cv2.imwrite(newpath, imgor)
                    train_new.append([img_path, i])
                else:
                    if np.random.randint(0, 4) < 3:
                        imgtr = transform(image=img)['image']
                        newpath = os.path.join(ROOT, f'{I}$' + img_path)
                        cv2.imwrite(newpath, imgtr)
                        train_new.append([f'{I}$' + img_path, i])

    if i in [7, 8]:
        for I in range(50):
            for img_path in tqdm(pre_dict[i]):
                transform = A.Compose(
                    [A.OneOf([
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                        A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
                        A.RGBShift(p=1),
                        A.ColorJitter(p=1),
                        A.FancyPCA(alpha=1, p=1)], p=1),
                        A.OneOf([A.HorizontalFlip(p=1),
                                 A.VerticalFlip(p=1),
                                 A.Rotate(p=1), ], p=1),
                        A.ShiftScaleRotate(p=1),
                        A.Resize(256, 256)])

                img = img_read(img_path)
                if I == 0:
                    imgor = A.Resize(256, 256)(image=img)['image']
                    newpath = os.path.join(ROOT, img_path)
                    cv2.imwrite(newpath, imgor)
                    train_new.append([img_path, i])
                else:
                    if np.random.randint(0, 9) < 7:
                        imgtr = transform(image=img)['image']
                        newpath = os.path.join(ROOT, f'{I}$' + img_path)
                        cv2.imwrite(newpath, imgtr)
                        train_new.append([f'{I}$' + img_path, i])
    if i in [5]:
        for I in range(60):
            for img_path in tqdm(pre_dict[i]):
                transform = A.Compose(
                    [A.OneOf([
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                        A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
                        A.RGBShift(p=1),
                        A.ColorJitter(p=1),
                        A.FancyPCA(alpha=1, p=1)], p=1),
                        A.OneOf([A.HorizontalFlip(p=1),
                                 A.VerticalFlip(p=1),
                                 A.Rotate(p=1), ], p=1),
                        A.ShiftScaleRotate(p=1),
                        A.Resize(256, 256)])

                img = img_read(img_path)
                if I == 0:
                    imgor = A.Resize(256, 256)(image=img)['image']
                    newpath = os.path.join(ROOT, img_path)
                    cv2.imwrite(newpath, imgor)
                    train_new.append([img_path, i])
                else:
                    if np.random.randint(0, 7) < 6:
                        imgtr = transform(image=img)['image']
                        newpath = os.path.join(ROOT, f'{I}$' + img_path)
                        cv2.imwrite(newpath, imgtr)
                        train_new.append([f'{I}$' + img_path, i])


dftrain = pd.DataFrame(train_new,columns=['image','label'])
dftrain
dftrain.to_csv('train.csv',index=False)
val_csv.to_csv('val.csv',index=False)




