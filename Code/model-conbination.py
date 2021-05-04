import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

from tqdm.notebook import tqdm

from PIL import Image
import os
import cv2

import torch
from torch import nn
from torchvision import transforms,models
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import albumentations as A
from albumentations.pytorch import ToTensorV2

df = pd.read_csv('eda-plant-pathology-2021/val.csv')
target_path = os.path.join('../input/val-aug' + "/*.jpg")
path_list = []

for path in glob.glob(target_path):
    path_list.append(path)


class ImageTransform():
    def __init__(self):
        self.data_transform = {
            'test': A.Compose([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                A.Resize(224, 224),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        }

    def __call__(self, img, phase="train"):
        img = np.array(img)
        return self.data_transform[phase](image=img)['image']


class PlantDataset(Dataset):

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        image_name = img_path[-20:]

        if self.phase in ["test"]:
            label = -1

        return img_transformed, label, image_name


test_dataset = PlantDataset(path_list, transform=ImageTransform(), phase='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
index = 1
print("\n【test dataset】")
print(f"img num : {test_dataset.__len__()}")
print(f"img : {test_dataset.__getitem__(index)[0].size()}")
print(f"label : {test_dataset.__getitem__(index)[1]}")
print(f"image name : {test_dataset.__getitem__(index)[2]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_pretrained = False

vggmodel = models.vgg16(pretrained=use_pretrained)
vggmodel.classifier[6] = nn.Linear(in_features=4096, out_features=12)

resmodel = models.resnet50(pretrained=use_pretrained)
resmodel.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 12)
)

vgg_path = "../input/vgg-v1/vgg16_fine_tuning_v1.h"
res_path = '../input/resnet/resnet18_fine_tuning_v1.h'
if torch.cuda.is_available():
    load_weights = torch.load(vgg_path)
    vggmodel.load_state_dict(load_weights)
    load_weights = torch.load(res_path)
    resmodel.load_state_dict(load_weights)
else:
    load_weights = torch.load(vgg_path, map_location={"cuda:0": "cpu"})
    vggmodel.load_state_dict(load_weights)
    load_weights = torch.load(res_path, map_location={"cuda:0": "cpu"})
    resmodel.load_state_dict(load_weights)

class_labels = ['complex', 'frog_eye_leaf_spot', 'frog_eye_leaf_spot complex', 'healthy',
                'powdery_mildew', 'powdery_mildew complex', 'rust', 'rust complex',
                'rust frog_eye_leaf_spot', 'scab', 'scab frog_eye_leaf_spot',
                'scab frog_eye_leaf_spot complex']


class PlantPredictor():

    def __init__(self, net, df_labels_idx, dataloaders_dict):
        self.net = net
        self.df_labels_idx = df_labels_idx
        self.loaders = dataloaders_dict
        self.df_submit = pd.DataFrame()
        self.ensemble = []
        self.dfname = []

    def __predict_max(self, out):
        maxid = np.argmax(out.detach().numpy(), axis=1)
        maxid = int(maxid)
        df_predicted_label_name = self.df_labels_idx[maxid]

        return df_predicted_label_name

    def outputarray(self, out):
        return out.detach().numpy()

    def inference(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Devices to be used : {device}")
        #      df_pred_list,dfname = [],[]
        for inputs, _, image_name in tqdm(self.loaders):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(device)
            inputs = inputs.to(device)
            out = self.net(inputs)
            device = torch.device("cpu")
            out = out.to(device)
            pred = self.outputarray(out)
            #          df_pred = self.__predict_max(out)
            self.dfname.append(image_name[0])
            self.ensemble.append(pred[0])
        #         df_pred_list.append(df_pred)
        self.ensemble = np.array(self.ensemble)
        #     self.df_submit = pd.DataFrame({'image': dfname,'labels':df_pred_list})

vpredictor = PlantPredictor(vggmodel, class_labels, test_dataloader)
vpredictor.inference()

rpredictor = PlantPredictor(resmodel, class_labels, test_dataloader)
rpredictor.inference()

datastack = np.hstack((vpredictor.ensemble,rpredictor.ensemble))

target = []
for la in vpredictor.dfname:
    zz=df[df.image == la].label
    target=target+zz.values.tolist()


x_train, x_val, y_train, y_val = train_test_split(datastack,target,
                                                      random_state = 42,
                                                      shuffle=True,
                                                      test_size=0.2,
                                                      stratify = target)
y_train, y_val = np.array(y_train),np.array( y_val)
print(f'the shape of train set:{x_train.shape},target {y_train.shape}')
print(f'the shape of val set:{x_val.shape},target {y_val.shape}')

clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5,
       max_depth=2, random_state=0).fit(x_train, y_train)
pred = clf.predict(x_val)

f1_score(pred,y_val,average='micro')

import pickle

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)


