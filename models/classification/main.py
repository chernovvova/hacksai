import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

# target = []
# flat_data = []
# images = []
DataDirectory = './datasets/'
#
Categories = [x[0] for x in os.walk(DataDirectory)]
# count = 0
# for i in Categories[1::]:
#   print("Category is:",i,"\tLabel encoded as:",Categories.index(i))
#   target_class = Categories.index(i)
#   path = os.path.join(i)
#   for img in os.listdir(path):
#     count += 1
#     if(count < 300):
#       img_array = imread(os.path.join(path,img))
#       img_resized = resize(img_array,(150,150,3))
#       flat_data.append(img_resized.flatten())
#       images.append(img_resized)
#       target.append(target_class)
#     else:
#       count = 0
#       break
# flat_data = np.array(flat_data)
# images = np.array(images)
# target = np.array(target)
#
#
# df = pd.DataFrame(flat_data)
# df['Target'] = target
#
#
# print(plt.imshow(images[20]))
#
#
# from sklearn.model_selection import train_test_split
#
# x = df.iloc[:,:-1].values
# y = target
# print("Input data dimensions:",x.shape)
# print("Output data dimensions:",y.shape)
# x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size = 0.3,random_state=109,stratify=y)
# print("Dimensions of input training data:",x_train.shape)
# print("Dimensions of input testing data:",x_test.shape)
# print("Dimensions of output training data:",y_train.shape)
# print("Dimensions of output testing data:",y_test.shape)
#
#
# print("Labels\t\t   Image index considered")
# print(np.unique(y_train,return_counts=True))
# print(np.unique(y_test,return_counts=True))
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
#
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                       'C': [1, 10, 100, 1000]}]
#
# cv = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=3)
# cv.fit(x_train, y_train)
#
#
# print("Best parameters to apply are:", cv.best_params_)
# svm = cv.best_estimator_
# print("Model after tuning is:\n",svm)
#
#
# y_prediction = svm.predict(x_test)
#
#
# print("Expected results: ",y_test)
# print("Predicted results:",y_prediction)
#
# from sklearn.metrics import f1_score
# print("f1_score:", f1_score(y_prediction, y_test, average='weighted'))
#
#
import pickle
# #{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
# print("Best parameters to apply are:", cv.best_params_)
#
# pickle.dump(svm,open("Classification_Model.p","wb"))
test_model = pickle.load(open("Classification_Model.p","rb"))

flat_data = []
url = "test/scale_1200.jpeg"
img_array = imread(url)
img_resized = resize(img_array,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)
print("Dimensions of original image are:", img_array.shape)
plt.imshow(img_resized)
y_output = test_model.predict(flat_data)
y_output = Categories[y_output[0]]
print("PREDICTED OUTPUT IS:",y_output)
