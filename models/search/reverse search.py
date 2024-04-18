import matplotlib.pyplot as plt
import os
import torch
import clip
from os import listdir
from os.path import splitext
import json
from PIL import Image
import pickle as pk

import time

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from string import punctuation
from pathlib import Path
directory = "./train"
#df = pd.read_csv('dataset/train.csv',sep=';')
# folder = []
# files = []
# imgmas = []
# for i in os.walk(directory):
#     folder.append(i[1])
#     files.append(i[2])
# folder = folder[0]
# files.pop(0)
# np_img = []
# image_path="./train_jpg"
# try:
#     for i in range(int(len(folder))):
#         for j in range(0,len(files[i])):
#             Image.open(directory+"/"+folder[i]+"/"+files[i][j]).convert('RGB').save(f"{image_path}/{files[i][j]}.jpg")
# except Exception as e:
#     print(f'{directory+"/"+folder[i]+"/"+files[i][j]}')
def show_images(images, figsize=(20,10), columns = 5):
  plt.figure(figsize=figsize)
  for i, image in enumerate(images):
      plt.subplot(int(len(images) / columns) + 1, columns, i + 1)
      plt.imshow(image)
plt.show()
IMAGES_PATH="./train_jpg"
file_names=os.listdir(IMAGES_PATH)
print(f"number of images: {len(file_names)}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32")
print(device)
def get_features(image):
    image =  preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def generate_clip_features():
    all_image_features=[]
    image_filenames=listdir(IMAGES_PATH)
    try:
       all_image_features=pk.load(open("clip_image_features.pkl", "rb"))
    except (OSError, IOError) as e:
       print("file_not_found")

    def exists_in_all_image_features(image_id):
        for image in all_image_features:
            if image['image_id'] == image_id:
                # print("skipping "+ str(image_id))
                return True
        return False

    def exists_in_image_folder(image_id):
        if image_id in image_filenames:
                return True
        return False

    def sync_clip_image_features():
        for_deletion=[]
        for i in range(len(all_image_features)):
            if not exists_in_image_folder(all_image_features[i]['image_id']):
                print("deleting "+ str(all_image_features[i]['image_id']))
                for_deletion.append(i)
        for i in reversed(for_deletion):
            del all_image_features[i]

    sync_clip_image_features()
    for image_filename in tqdm(image_filenames):
        image_id=splitext(image_filename)[0]
        if exists_in_all_image_features(image_id):
            continue
        image=Image.open(IMAGES_PATH+"/"+image_filename)
        image_features=get_features(image)
        all_image_features.append({'image_id':image_id,'features':image_features})
    pk.dump(all_image_features, open("clip_image_features.pkl","wb"))
# directory = "./test"
# folder = []
# files = []
# imgmas = []
# for i in os.walk(directory):
#     folder.append(i[1])
#     files.append(i[2])
# folder = folder[0]
# files.pop(0)
# np_img = []
# image_path="./test_jpg"
# try:
#     for i in range(int(len(folder))):
#         for j in range(0,len(files[i])):
#             Image.open(directory+"/"+folder[i]+"/"+files[i][j]).convert('RGB').save(f"{image_path}/{files[i][j]}")
# except Exception as e:
#     print(f'{directory+"/"+folder[i]+"/"+files[i][j]}')
#generate_clip_features()

query_image_pillow=Image.open(f'200661.jpg')
query_image_features=get_features(query_image_pillow)
show_images([np.array(query_image_pillow)])
print(query_image_features.shape)
image_features=pk.load(open("clip_image_features.pkl", "rb"))
features=[]
for image in image_features:
    features.append(np.array(image['features']))
features=np.array(features)
features=np.squeeze(features)
# print(features.shape)
# exit()
path="./train_jpg"
start=time.time()
knn = NearestNeighbors(n_neighbors=20,algorithm='brute',metric='cosine') #euclidean
knn.fit(features)
file_names=listdir(path)

distances, indices = knn.kneighbors(query_image_features, return_distance=True)
finish=time.time()
print("Время работы KNN: ", finish-start)

found_images=[]
path_images=[]
dictionary = dict()
i=0
df = pd.read_csv("./train.csv",sep=";")
text =''
fig = plt.figure(figsize=(7,5))
for x in indices[0]:
    found_images.append(np.array(Image.open(path+"/"+file_names[x])))
    if(type(df.loc[df['img_name'] == file_names[x].replace(".jpg","",1), 'description'].iloc[0]) == str):
        text += df.loc[df['img_name'] == file_names[x].replace(".jpg", "", 1), 'description'].iloc[0]
        text+="\n"
    path_images.append(file_names[x].replace(".jpg","",1))
    ax = fig.add_subplot(5, 4, i+1)
    plt.imshow(np.array(Image.open(path+"/"+file_names[x])))
    plt.axis('off')
    ax.set_title(f'{i+1}-ая image')
    dictionary.update({f'{file_names[x].replace(".jpg","",1)}': distances[0][i]})
    i+=1
plt.show()
text = text.lower()

text = text.translate(str.maketrans("", "", punctuation))

n_chars = len(text)
print(path_images)
print(n_chars)
print(text)
x=[f"{i}" for i in range(len(indices[0]))]
y=[]
for i in distances[0]:
    y.append(i)
print(y)
print(dictionary)
# plt.xlabel('Похожие картинки')
# plt.ylabel('Расстояние в метрическом пространстве')
# plt.axis([-0.5,len(indices[0])-0.5,0,0.5])
# plt.bar(x,y)
# plt.show()
# # #show_images(np.array(found_images))