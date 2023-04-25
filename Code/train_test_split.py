import pandas as pd
from sklearn.model_selection import train_test_split
import os

path = "/home/ubuntu/Final_Project"
folders = os.listdir(path + '/Data')
nclasses = len(folders)

image_paths = []
labels = []
targets=[]
for i in range(len(folders)):
    image_names = os.listdir(path + '/Data/' + folders[i])
    image_paths1 = [path + '/Data/' + folders[i] + '/' + j for j in image_names]
    labels1=[folders[i] for j in range(len(image_paths1))]
    targets1=[i]*len(image_paths1)
    image_paths.extend(image_paths1)
    labels.extend(labels1)
    targets.extend(targets1)
del image_names
del image_paths1
del labels1
del targets1

image_paths_train, image_paths_test, labels_train, labels_test = train_test_split(image_paths, labels, test_size=0.3, random_state=42, stratify=labels)
image_paths_validate, image_paths_test, labels_validate, labels_test = train_test_split(image_paths_test, labels_test, test_size=0.5, random_state=42, stratify=labels_test)
len(image_paths_train), len(image_paths_test), len(labels_train), len(labels_test), len(image_paths_validate), len(labels_validate)
list1 = pd.concat([pd.DataFrame({'path': image_paths_train, 'label': labels_train, 'split':'train'}),
pd.DataFrame({'path': image_paths_test, 'label': labels_test, 'split':'test'}),
pd.DataFrame({'path': image_paths_validate, 'label': labels_validate, 'split':'validate'})])
classes=list(set(list1.label))
classcode = [i for i in range(len(classes))]
classdict = pd.DataFrame({'label':classes, 'target':classcode}).set_index('label').to_dict('index')
list1['target'] = [classdict[i]['target'] for i in list1.label]
list1.to_excel(path + '/' + 'image_paths.xlsx', index=False)