import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from tqdm.notebook import tqdm
from PIL import Image
import os
import cv2

import torch
from torch import nn
from torchvision import transforms,models
from torch.utils.data import Dataset, DataLoader ,random_split
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

df=pd.read_csv('/kaggle/input/plant-pathology-2021-fgvc8/train.csv')
df.labels=df.labels.apply(lambda x:x.split(' '))
class_labels=list(set(df.labels.sum()))
class_labels

def label_exchange(yarray):

  le=LabelEncoder()
  le.fit(['powdery_mildew', 'rust', 'frog_eye_leaf_spot', 'scab', 'healthy', 'complex'])
  output=[]

  for label in yarray:
    trans=le.transform(label)
    y=torch.zeros(6, dtype=torch.long).scatter_(dim=0, index=torch.tensor(trans), value=1)
    output.append(y.numpy())
  return output,le.classes_

target, class_labels = label_exchange(df.labels)

for label in class_labels:
  df[label] = 0

df.iloc[:,2:] = target
del(df['labels'])

#-----------parameter------------
BATCH_SIZE = 128
EPOCHS = 4
SEED = 42
DROPOUT = 0.2
THRESHOLD = 0.5
LR = 1e-4
IMG_DIR = '../input/plant-pathology-2021-fgvc8/train_images'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reference https://www.kaggle.com/kuboko/pp2021-pytorch-vgg-16-fine-tune-inference
class pl_transform():
    def __init__(self):
        self.plant_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, img):
        return self.plant_transform(img)

# imgpath = '../input/plant-pathology-2021-fgvc8/train_images/803b586d7db3ca16.jpg'
# image1 =  Image.open(imgpath)
# ptransform = pl_transform()
# zz=ptransform(image1)
# zz=zz.numpy().transpose([1, 2, 0])
# plt.imshow(zz)
# plt.show()

class mydataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.targetfile = csv_file
        self.root = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.targetfile)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.targetfile.iloc[idx, 0])
        image = Image.open(img_path)
        label = torch.tensor(self.targetfile.iloc[idx, 1:].tolist(), dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)
        return image, label

mydata =  mydataset(df , IMG_DIR , transforms = pl_transform())
valid_no = int(len(mydata)*0.2)
train_set ,val_set  = random_split(mydata , [len(mydata) - valid_no  ,valid_no],torch.Generator().manual_seed(SEED))
print(f"train_set len {len(train_set)} val_set len {len(val_set)}")
loader = {"train":DataLoader(train_set , shuffle=True , batch_size=BATCH_SIZE),
              "val": DataLoader(val_set , shuffle=True , batch_size=BATCH_SIZE)}

# # Operation check
# batch_iterator = iter(loader["train"])
# inputs, labels = next(batch_iterator)
# print(inputs.size())  # torch.Size([3, 3, 224, 224]) : [batch_size, Channel, H, W]
# print(labels)

use_pretrained = False
Mymodel = models.vgg16(pretrained=use_pretrained)

Mymodel.classifier[6] =  nn.Linear(in_features=4096, out_features=6)
# save_path = "/kaggle/working/vgg16_pretrained.h"
# torch.save(Mymodel.state_dict(), save_path)

load_path = "../input/model-plant-pathology-2021/vgg16_fine_tuning_v1.h"
if torch.cuda.is_available():
    load_weights = torch.load(load_path)
    Mymodel.load_state_dict(load_weights)
else:
    load_weights = torch.load(load_path, map_location={"cuda:0": "cpu"})
    Mymodel.load_state_dict(load_weights)

Mymodel.train()

# Store the parameters to be learned by finetuning in the variable params_to_update.
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# Specify the parameter name of the layer to be trained.
update_param_names_1 = ["features.24.weight", "features.24.bias", "features.26.weight", "features.26.bias", "features.28.weight", "features.28.bias"]
update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

for name, param in Mymodel.named_parameters():
    if name in update_param_names_1:
        param.requires_grad = True
        params_to_update_1.append(param)
        print(f"Store in params_to_update_1 : {name}")
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print(f"Store in params_to_update_2 : {name}")
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print(f"Store in params_to_update_3 : {name}")
    else:
        param.requires_grad = False
        print(f"Parameters not to be learned :  {name}")

# ------------op&loss-----------
Mymodel.to(device)

loss_fn=nn.BCEWithLogitsLoss(reduction='sum')

optimizer = torch.optim.Adam([
    {"params": params_to_update_1, "lr": 1e-4},
    {"params": params_to_update_2, "lr": 5e-4},
    {"params": params_to_update_3, "lr": 1e-3}
])

def multi_result(pred ,THRESHOLD):
  pred=pred.cpu().detach().numpy()
  z=pred>=THRESHOLD
  z=z.astype('float32')
  return z


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Devices to be used : {device}")
    net.to(device)
    torch.backends.cudnn.benchmark = True
    # loop for epoch
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
            # if (epoch == 0) and (phase == "train"):
            # continue
            for inputs, labels in tqdm(loader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    fit_target = multi_result(outputs, THRESHOLD)
                    num = np.all(labels.cpu().numpy() == fit_target, axis=1).sum()
                    # print(num)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item()
                    epoch_corrects += num
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloaders_dict[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


# train_model(Mymodel, loader, loss_fn, optimizer, num_epochs=EPOCHS)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Devices to be used : {device}")
# for t in range(EPOCHS):
#     print(f'Epoch {t + 1}:\n-----------------')
#     count=0
#     for x, y in tqdm(loader['train']):
#         count+=1
#         size = len(loader['train'].dataset)

#         x, y = x.to(device), y.to(device)

#         # compute error
#         pred = Mymodel(x)
#         loss = loss_fn(pred, y)

#         # back
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if count % 10 == 0:
#             loss, current = loss.item()/size, count * len(x)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#         Mymodel.eval()
#         with torch.no_grad():
#             test_loss, correct = 0, 0
#             for x, y in loader['val']:
#                 x, y = x.to(device), y.to(device)

#                 pred = Mymodel(x)
#                 test_loss += loss_fn(pred, y).item()
#                 fit_target = multi_result(pred, THRESHOLD)

#                 num = np.all(y.cpu().numpy() == fit_target, axis=1).sum()
#                 correct += num

#     test_loss /= len(loader['val'].dataset)
#     correct /= len(loader['val'].dataset)


#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# save_path = "./vgg16_fine_tuning_v1.h"
# torch.save(Mymodel.state_dict(), save_path)

def f1(loader):
    Mymodel.eval()
    with torch.no_grad():
        for x, y in tqdm(loader['val']):
            x, y = x.to(device), y.to(device)
            pred = Mymodel(x)
            fit_target = multi_result(pred, THRESHOLD)
            print(pred,fit_target)
            print("F1 score", f1_score(pred,fit_target, average = 'macro'))

f1(loader)

submission=pd.read_csv('../input/plant-pathology-2021-fgvc8/sample_submission.csv')
TEST_DIR='/kaggle/input/plant-pathology-2021-fgvc8/test_images/'


def predict(submissionfile):
    images = []

    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    for img_path in submissionfile.image:
        img_path = os.path.join(TEST_DIR, img_path)
        input_image = Image.open(img_path)
        images.append(np.array(trans(input_image)))

    x = torch.FloatTensor(np.array(images)).to(device)

    with torch.no_grad():
        y_pred = Mymodel(x)
    y_pred = multi_result(y_pred, THRESHOLD)

    return y_pred


def labelconvert(lst):
    output = []
    for target in lst:
        ta = []
        for i in range(6):
            if target[i] != 0:
                ta.append(class_labels[i])
        output.append(ta)

    return (output)

submission.labels = labelconvert(predict(submission))
delimiter = ' '
submission.labels = submission.labels.apply(lambda x: delimiter.join(x) )

def emm(x):
    if x=='':
        x='healthy'
    return x
submission.labels = submission.labels.apply(lambda x: emm(x))
submission


imgpath = '../input/plant-pathology-2021-fgvc8/test_images/85f8cb619c66b863.jpg'
image1 =  Image.open(imgpath)
ptransform = trans = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
zz=ptransform(image1)
zz=zz.numpy().transpose([1, 2, 0])
plt.imshow(image1)
plt.show()
plt.imshow(zz)
plt.show()

submission.to_csv('submission.csv',index=False)

