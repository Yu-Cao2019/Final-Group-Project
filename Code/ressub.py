import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tqdm.notebook import tqdm

from PIL import Image
import os
import cv2

import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader


import albumentations as A
from albumentations.pytorch import ToTensorV2
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

TEST_DIR = '../input/plant-pathology-2021-fgvc8/test_images'
target_path = os.path.join(TEST_DIR + "/*.jpg")
path_list = []

for path in glob.glob(target_path):
    path_list.append(path)


class ImageTransform():
    def __init__(self):
        self.plant_transform = {
            'test': A.Compose([
                A.Resize(224, 224),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        }

    def __call__(self, img, phase="test"):
        img = np.array(img)
        return self.plant_transform[phase](image=img)['image']


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
Mymodel = models.resnet50(pretrained=use_pretrained)
Mymodel.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 12)
        )

load_path = "../input/resnet/resnet18_fine_tuning_v1.h"
if torch.cuda.is_available():
    load_weights = torch.load(load_path)
    Mymodel.load_state_dict(load_weights)
else:
    load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})
    Mymodel.load_state_dict(load_weights)

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

    def __predict_max(self, out):
        maxid = np.argmax(out.detach().numpy(), axis=1)
        maxid = int(maxid)
        df_predicted_label_name = self.df_labels_idx[maxid]

        return df_predicted_label_name

    def inference(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Devices to be used : {device}")
        df_pred_list, dfname = [], []
        for inputs, _, image_name in tqdm(self.loaders):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(device)
            inputs = inputs.to(device)
            out = self.net(inputs)
            device = torch.device("cpu")
            out = out.to(device)
            df_pred = self.__predict_max(out)
            dfname.append(image_name[0])
            df_pred_list.append(df_pred)

        self.df_submit = pd.DataFrame({'image': dfname, 'labels': df_pred_list})

predictor = PlantPredictor(Mymodel, class_labels, test_dataloader)
predictor.inference()

# Mymodel.to(device)
# def predict(dir):
#     images = []
#     lst = []
#     trans = A.Compose([
#                 A.Resize(256,256),
#                 A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=1),
#                 A.CenterCrop(224,224),
#                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ToTensorV2(),
#             ])

#     for path in os.listdir(dir):
#         img_path = os.path.join(TEST_DIR,path)
#         input_image = Image.open(img_path)
#         input_image = np.array(input_image)
#         images.append(np.array(trans(image = input_image)['image']))
#         lst.append(path)


#     x = torch.FloatTensor(np.array(images)).to(device)

#     with torch.no_grad():
#         y_pred = Mymodel(x)
#         device = torch.device('cpu')
#         y_pred = y_pred.to(device)
#         y_pred = np.argmax(y_pred.detach().numpy(),axis = 1)

#     return lst,y_pred
# lst,ll = predict(TEST_DIR)

df_submit = predictor.df_submit.copy()

df_submit.to_csv('submission.csv',index=False)
