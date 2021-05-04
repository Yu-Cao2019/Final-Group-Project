## Code
This folder includs nine file -- 'eda-plant-pathology-2021.ipynb', 'failed-ensemble.ipynb', 'model-conbination.ipynb', 'model-plant-pathology-2021.ipynb', 'new-eda.ipynb', 'resnet.ipynb', 'ressub.ipynb', 'val-aug.ipynb' and 'vgg-v1.ipynb'.  
*Note*: Because the dataset is too large, we didn't download it and just wrote code in kaggle and run them. So, if you want to run our code, maybe you can use 'kaggle competitions download
```
-c plant-pathology-2021-fgvc8' to download the dataset.  
```
***
If you want to execute this code, please follow this process:
```
eda-plant-pathology-2021.ipynb
new-eda.ipynb
```
These two are used to data expansion. The first is the version of pixel+spatial transform and the second is the only spatial level transform.After downloading the data from kaggle, you can execute these two first.  
```
resnet.ipynb
vgg-v1.ipynb
```
These two notebook are used to execute the resnet model and vgg model  
```
model-conbination.ipynb
```
This notebook combines the vgg ang resnet using ensemble model.
```
failed-ensemble.ipynb
```
This notebook is for final submission
