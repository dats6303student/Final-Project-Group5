import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import numpy
import random
from datetime import datetime
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--split", default=None, type=str, required=True)  
parser.add_argument("--split_by_contrast", default=None, type=str, required=True)  
parser.add_argument("--contrast", default=None, type=str, required=True)  
parser.add_argument("--batchsize", default=None, type=str, required=True)  
args = parser.parse_args()
print(args)

batch_size = int(args.batchsize)
SPLIT = args.split
CONTRAST=args.contrast
SPLIT_BY_CONTRAST=args.split_by_contrast

path = "/home/ubuntu/Final_Project"

image_paths_df = pd.read_excel(path + '/' + 'image_paths.xlsx')
nclasses=len(set(image_paths_df['target']))
target_dict = image_paths_df[['target', 'label']].drop_duplicates()
target_dict = target_dict.set_index('target').to_dict('index')

def f(txt):
    if txt.find("T1C+")>-1: return 'T1C+'
    if txt.find("T1")>-1: return 'T1'
    if txt.find("T2")>-1: return 'T2'
image_paths_df['contrast'] = [f(label) for label in list(image_paths_df['label'])]
image_paths_df = image_paths_df[image_paths_df["split"]==SPLIT]
if SPLIT_BY_CONTRAST=='Y': image_paths_df = image_paths_df[image_paths_df["contrast"]==CONTRAST]

image_paths = list(image_paths_df['path'])
targets = list(image_paths_df['target'])

image_size=630
def process_image(img):
    name=img
    img = cv2.imread(img)
    img= cv2.resize(img,(image_size,image_size))
    img.shape = (3,image_size,image_size)
    return img  
def recode(Y):
    y=numpy.zeros((Y.shape))
    argmax1 = numpy.argmax(Y,axis=1)
    row=-1
    for i in argmax1:
        row+=1
        y[row][i]=1
    return y

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, nclasses)
        
    def forward(self, x):
        return self.resnet(x)

model = Resnet18()
model.load_state_dict(torch.load('model.pt'))

model.eval()
print('Evaluation mode')
for i in range(0, len(image_paths), batch_size):
    j = i+batch_size
    print('Processing images {a} to {b} out of {c}'.format(a=i+1,b=j,c=len(image_paths)))
    image_paths1 = image_paths[i:j]
    testX = numpy.array([process_image(i) for i in image_paths1])
    Xtest=torch.tensor(testX, dtype=torch.float)
    new = model(Xtest)
    new = new.detach().numpy()
    if i==0: Yfitted_test= new
    else: Yfitted_test = numpy.vstack((Yfitted_test,new))
Ytest=torch.tensor(targets, dtype=torch.long)
Yfitted_test = numpy.argmax(Yfitted_test,axis=1)
Ytest = [target_dict[i.item()]['label'] for i in Ytest]
Yfitted_test = [target_dict[i]['label'] for i in Yfitted_test]
print(accuracy_score(Ytest,Yfitted_test))
print(classification_report(Ytest,Yfitted_test))
	
  