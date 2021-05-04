import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tqdm.notebook import tqdm
from PIL import Image
import os
import cv2

import torch
from torch import nn
from torchvision import transforms,models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

df=pd.read_csv('../input/new-eda/train.csv')
df.label.value_counts()

for _ in range(5):
    df = df.sample(frac=1)

#-----------parameter------------
SEED = 42
EPOCHS = 6
LR = 1e-5
MIN_LR = 1e-7
MODE = 'min'
FACTOR = 0.2
PATIENCE = 0
BATCH_SIZE = 128
TEST_SIZE = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def data_split(phase = 'train',size =0.2):
#     x_train, x_val, y_train, y_val = train_test_split(df.image,df.label_ex,
#                                                       random_state = SEED,
#                                                       shuffle=True,
#                                                       test_size=size,
#                                                       stratify =df.label_ex)
#     tar_csv = pd.DataFrame()
#     if phase in ['train']:
#         tar_csv['image'] = x_train
#         tar_csv['label'] = y_train
#     elif phase in ['val']:
#         tar_csv['image'] = x_val
#         tar_csv['label'] = y_val
#     elif phase in ['test']:
#         DIR = '../input/plant-pathology-2021-fgvc8/sample_submission.csv'
#         tar_csv = pd.read_csv(DIR)

#     return tar_csv

train_csv = df
val_csv = pd.read_csv('../input/new-eda/val.csv')
print(f'The test size is {TEST_SIZE}\nThe length of train set is {len(train_csv)}')
print(f'The length of validation set is {len(val_csv)}')

train_csv.label.value_counts()

# def train_weight(train_csv):
#     numarr = train_csv.label.value_counts().sort_index().values
#     weights = 1.0/torch.tensor(np.log(numarr),dtype = torch.float)
#     train_target = train_csv.label.tolist()
#     sample_weights = weights[train_target]
#     return sample_weights

class pl_transform():
    def __init__(self):
        self.plant_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    224, scale=(0.5, 1.0)),
                #                 transforms.RandomHorizontalFlip(),
                #                 transforms.RandomVerticalFlip(0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),  # ([0.47955528, 0.6252535, 0.4016591], [0.1559643, 0.13600954, 0.16537014])

            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }

    def __call__(self, img, phase='train'):
        return self.plant_transform[phase](img)


#         else:
#             img =np.array(img)
#             return self.plant_transform[phase](image = img)['image']

class mydataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None, phase='train'):
        self.targetfile = csv_file
        self.root = img_dir
        self.transforms = transforms
        self.phase = phase

    def __len__(self):
        return len(self.targetfile)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.targetfile.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.targetfile.iloc[idx, 1]

        if self.transforms:
            image = self.transforms(image, self.phase)
        return image, label


ROOT_TRAIN = '../input/new-eda/aug_re_img'
ROOT_VAL = '../input/resized-plant2021/img_sz_256'
train_dataset = mydataset(train_csv, ROOT_TRAIN, pl_transform())
val_dataset = mydataset(val_csv, ROOT_VAL, pl_transform(), phase='val')

index = 0

print("【train dataset】")
print(f"img num : {train_dataset.__len__()}")
print(f"img : {train_dataset.__getitem__(index)[0].size()}")
print(f"label : {train_dataset.__getitem__(index)[1]}")
print("\n【validation dataset】")
print(f"img num : {val_dataset.__len__()}")
print(f"img : {val_dataset.__getitem__(index)[0].size()}")
print(f"label : {val_dataset.__getitem__(index)[1]}")

# tweights = train_weight(train_csv)
# wsampler = WeightedRandomSampler(weights=tweights, num_samples=len(tweights), replacement=True)
train_loader = DataLoader(train_dataset,
#                        sampler = wsampler,
                          batch_size = BATCH_SIZE,
                          shuffle = True,
                         )
val_loader = DataLoader(val_dataset,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                         )

loader = {"train": train_loader, "val": val_loader}


def model_define():
    use_pretrained = True
    Mymodel = models.resnet18(pretrained=use_pretrained)
    Mymodel.fc = nn.Sequential(
        #             nn.Linear(2048, 1024),
        #             nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 12)
    )

    params_to_update_1 = []
    update_param_names_1 = ['fc.1.weight', 'fc.1.bias']
    for name, param in Mymodel.named_parameters():
        if name in update_param_names_1:
            param.requires_grad = True
            params_to_update_1.append(param)
            print(f"Store in params_to_update_1 : {name}")
        elif name[5:8] in ['4.1']:
            param.requires_grad = True
            params_to_update_1.append(param)
            print(f"Store in params_to_update_1 : {name}")
        else:
            param.requires_grad = False
            print(f"Parameters not to be learned :  {name}")

    Mymodel.to(device)
    return params_to_update_1, Mymodel


params_to_update_1, Mymodel = model_define()
Mymodel.train()

loss_fn=nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {"params": params_to_update_1}] , lr =LR)

sgdr_partial = lr_scheduler.StepLR(optimizer, step_size =4, gamma=0.1 )


def train_model(net, loader, criterion, optimizer, sgdr_partial, num_epochs):
    """
    Function for training the model.

    Parameters
    ----------
    net: object
    dataloaders_dict: dictionary
    criterion: object
    optimizer: object
    num_epochs: int
    """
    print(f"Devices to be used : {device}")
    torch.backends.cudnn.benchmark = True
    # loop for epoch
    train_acc = []
    val_acc = []

    train_loss = []
    val_loss = []

    lr = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")
        print("-------------------------------")
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0
            #             if (epoch == 0) and (phase == "train"):
            #                 continue
            f1lst_pred = []
            f1lst_true = []
            for inputs, labels in tqdm(loader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'val':
                        f1lst_pred += preds.data.tolist()
                        f1lst_true += labels.data.tolist()
                    # print(num)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        # sgdr_partial.step()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(loader[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(loader[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == 'train':
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            #                 train_acc += epoch_acc.tolist()
            #                 train_loss += epoch_loss.tolist()

            if phase == 'val':
                print(f"The f1 score is {f1_score(f1lst_pred, f1lst_true, average='weighted')}")
                print(f"Learning Rate is {optimizer.param_groups[0]['lr']}")
                sgdr_partial.step()
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)
                lr.append(optimizer.param_groups[0]['lr'])
    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_acc, color='r', marker='o', label='train/acc')
    plt.plot(val_acc, color='b', marker='x', label='val/acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.show()

    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_loss, color='r', marker='o', label='train/loss')
    plt.plot(val_loss, color='b', marker='x', label='val/loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.show()

    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(lr, color='g', marker='o', label='learning rate')
    plt.ylabel('LR')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.show()


train_model(Mymodel, loader, loss_fn, optimizer,sgdr_partial, EPOCHS)

save_path = "./resnet18_fine_tuning_v1.h"
torch.save(Mymodel.state_dict(), save_path)

# root = '../input/resized-plant2021/img_sz_384'
# for img in os.listdir(root):

#     img_path = os.path.join(root,img)
#     image = cv2.imread(img_path)
#     #image=np.array(image)
#     print(image.shape)


