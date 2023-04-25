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
import datetime
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--use_saved_weights", default=None, type=str, required=True) 
parser.add_argument("--nepochs", default=None, type=str, required=True)  
parser.add_argument("--batchsize", default=None, type=str, required=True)  
args = parser.parse_args()

load = args.use_saved_weights
nepochs = int(args.nepochs)
batch_size = int(args.batchsize)

path = "/home/ubuntu/Final_Project"

image_paths_df = pd.read_excel(path + '/' + 'image_paths.xlsx')

image_paths_df1 = image_paths_df[image_paths_df["split"]=='train']
image_paths_df2 = image_paths_df[image_paths_df["split"]=='test']
image_paths_df3 = image_paths_df[image_paths_df["split"]=='validate']

nclasses=len(set(image_paths_df['target']))

image_paths = list(image_paths_df1['path'])
targets = list(image_paths_df1['target'])
image_paths_test = list(image_paths_df2['path'])
targets_test = list(image_paths_df2['target'])
image_paths_validate = list(image_paths_df3['path'])
targets_validate = list(image_paths_df3['target'])

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
if load=='Y': 
    print('loading saved weights from model.pt')
    model.load_state_dict(torch.load('model.pt'))
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#loss_func = torch.nn.BCELoss()
loss_func=torch.nn.CrossEntropyLoss()

for epoch in range(nepochs):
    model.train()
    print("Training mode")
    ind = [i for i in range(len(image_paths))]
    random.shuffle(ind)
    image_paths_shuffled = [image_paths[i] for i in ind]
    targets_shuffled = [targets[i] for i in ind]
    for i in range(0, len(image_paths), batch_size):
        j = i+batch_size
        print('Processing images {a} to {b} out of {c} in epoch {e}'.format(a=i+1,b=j,c=len(image_paths), e=epoch+1))        
        image_paths1 = image_paths_shuffled[i:j]
        trainX = numpy.array([process_image(i) for i in image_paths1])
        trainY = targets_shuffled[i:j]
        inputs=torch.tensor(trainX, dtype=torch.float)
        outputs=torch.tensor(trainY, dtype=torch.long)
        optimizer.zero_grad()
        results=model(inputs)
        loss = loss_func(results,outputs)
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), "model.pt")