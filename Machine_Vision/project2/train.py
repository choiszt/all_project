import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import argparse
import tqdm
import logging
import copy
parser = argparse.ArgumentParser(description='KNN hyperparameters')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for data loader')
parser.add_argument("--alldata",type=bool,default=True)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
print(f"device:{device}")
import torchvision.transforms as transforms
class_names=[name.split('/')[-1] for name in glob.glob("/mnt/ve_share/liushuai/cv2/data/train/*")]
# print(class_names)
class ImageDataset(Dataset):
	def __init__(self,filepath,transform=lambda img:cv2.resize(img,(256,256))):
		self.filepath=filepath
		self.transform=transform
		with open(filepath,'r')as f:
			self.datalist=f.readlines()
	def __len__(self):
		return len(self.datalist)
	def __getitem__(self, index):
		file=self.datalist[index].strip()
		image=cv2.imread(file,flags=0)
		image=self.transform(image)
		# print(image.shape)
		label=file.split("/")[-2]
		# label=label.to(device)
		return image,label
trainpath='/mnt/ve_share/liushuai/cv2/data/train.txt'
testpath="/mnt/ve_share/liushuai/cv2/data/test.txt"
traindataset=ImageDataset(filepath=trainpath)
testdataset=ImageDataset(filepath=testpath)
args.batch_size=len(traindataset)
traindataloader=DataLoader(traindataset,batch_size=args.batch_size,shuffle=True,num_workers=8)
testdataloader=DataLoader(testdataset,batch_size=len(testdataset),shuffle=True,num_workers=8)
#-------------loading data--------------#
if args.alldata:
    for data,label in tqdm.tqdm(traindataloader,desc="loading train_data..."):
        train_data=data
        train_label=label
    for data,label in tqdm.tqdm(testdataloader,desc="loading test_data..."):
        test_data=data
        test_label=label
from sklearn.neighbors import KNeighborsClassifier
# train model
def trainKNN(data, labels, k,task=None):
	neigh = KNeighborsClassifier(n_neighbors=k, p=2)
	if task=='task1':
		flattendata=data.view(len(data),-1)
		neigh.fit(flattendata, labels) 
	neigh.fit(data, labels)           
	return neigh
# cluster = [5, 10, 15, 20, 25, 30]
# logging.basicConfig(filename='knn_results.log', level=logging.INFO,filemode='a')
# logging.info("--------------------")
# logging.info(f"batch_size={args.batch_size}")
# for i in range(len(cluster)):
# 	print(f'cluster={cluster[i]},  start training KNN...')
# 	model = trainKNN(train_data, train_label, cluster[i],task='task1')
# 	print("start testing KNN...")
# 	flatten_test_data=test_data.view(len(test_data),-1) #将后两维reshape flatten成一维
# 	predict_test = model.predict(flatten_test_data)
# 	print ("k =", cluster[i], ", Accuracy: ", np.mean(predict_test == label)*100, "%")
# 	logging.info(f"k = {cluster[i]}, Mean accuracy: {np.mean(predict_test == label)*100:.2f}")


#---------------------task 2: using SIFT+kmeans--------------------------#
from sklearn.cluster import KMeans
from sklearn import preprocessing
def computeSIFT(data):
    x = []
    for i in range(0, len(data)):
        sift = cv2.SIFT_create()
        img = data[i]
        step_size = 15
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        dense_feat = sift.compute(img, kp)
        x.append(dense_feat[1])
        
    return x

# extract dense sift features from training images
convert_to_np=lambda data:np.array(data) #convert data format from tensor to ndarray
train_data=convert_to_np(train_data)
test_data=convert_to_np(test_data)
x_train = computeSIFT(train_data) #(200,267)转换成 (252*128)
x_test = computeSIFT(test_data)

all_train_desc = []
for i in range(len(x_train)):
    for j in range(x_train[i].shape[0]):
        all_train_desc.append(x_train[i][j,:])

all_train_desc = np.array(all_train_desc) #729000*128

# build BoW presentation from SIFT of training images 
def clusterFeatures(all_train_desc, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_train_desc)
    return kmeans

# form training set histograms for each training image using BoW representation
def formTrainingSetHistogram(x_train, kmeans, k):
    train_hist = []
    for i in range(len(x_train)):
        data = copy.deepcopy(x_train[i])
        predict = kmeans.predict(data) #252
        train_hist.append(np.bincount(predict, minlength=k).reshape(1,-1).ravel())
        
    return np.array(train_hist)


# build histograms for test set and predict
def predictKMeans(kmeans, scaler, x_test, train_hist, train_label, k):
    # form histograms for test set as test data
    test_hist = formTrainingSetHistogram(x_test, kmeans, k)
    #vishist(test_hist) #visualization
    # make testing histograms zero mean and unit variance
    test_hist = scaler.transform(test_hist)
    
    # Train model using KNN
    knn = trainKNN(train_hist, train_label, k)
    predict = knn.predict(test_hist)
    return np.array([predict], dtype=np.array([test_label]).dtype)
    
def vishist(train_hist):
    fig, ax = plt.subplots()
    total=train_hist[0].copy()
    for i in range(1,train_hist.shape[0]):
        total+=train_hist[i]
    ax.bar(np.arange(train_hist.shape[1]), total, width=0.2)
# 设置图例和标签
    ax.legend()
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.set_title("trainhist")
    plt.savefig(f"{str(train_hist)}.jpg")

#---------------------task 2: implementing--------------------------#
# k = [10, 15, 20, 25, 30, 35, 40]
# logging.basicConfig(filename='sift_knn_results.log', level=logging.INFO,filemode='a')
# logging.info("--------------------")
# logging.info(f"batch_size={args.batch_size}")
# for i in range(len(k)):
#     kmeans = clusterFeatures(all_train_desc, k[i])
#     train_hist = formTrainingSetHistogram(x_train, kmeans, k[i])
#     vishist(train_hist)
#     # preprocess training histograms
#     scaler = preprocessing.StandardScaler().fit(train_hist)
#     train_hist = scaler.transform(train_hist) #标准化处理 缩放到0-1区间
    
#     predict = predictKMeans(kmeans, scaler, x_test, train_hist, train_label, k[i])
#     accuracy=lambda predict_label,test_label:np.mean(np.array(predict_label.tolist()[0]) == np.array(test_label))
#     res = accuracy(predict, test_label)
#     print("k =", k[i], ", Accuracy:", res*100, "%")
#     logging.info(f"k = {k[i]}, Mean accuracy: {res*100:.2f}")

#---------------------task 3: bag of sift+SVM--------------------------#
from sklearn.svm import LinearSVC
k = 60
kmeans = clusterFeatures(all_train_desc, k)

# form training and testing histograms
train_hist = formTrainingSetHistogram(x_train, kmeans, k)
test_hist = formTrainingSetHistogram(x_test, kmeans, k)

# normalize histograms
scaler = preprocessing.StandardScaler().fit(train_hist)
train_hist = scaler.transform(train_hist)
test_hist = scaler.transform(test_hist)

for i in range(len(k)):
    kmeans = clusterFeatures(all_train_desc, k[i])
    train_hist = formTrainingSetHistogram(x_train, kmeans, k[i])
    vishist(train_hist)
    # preprocess training histograms
    scaler = preprocessing.StandardScaler().fit(train_hist)
    train_hist = scaler.transform(train_hist) #标准化处理 缩放到0-1区间
    
    predict = predictKMeans(kmeans, scaler, x_test, train_hist, train_label, k[i])
    accuracy=lambda predict_label,test_label:np.mean(np.array(predict_label.tolist()[0]) == np.array(test_label))
    res = accuracy(predict, test_label)
    print("k =", k[i], ", Accuracy:", res*100, "%")
    logging.info(f"k = {k[i]}, Mean accuracy: {res*100:.2f}")
logging.basicConfig(filename='sift_knn_results.log', level=logging.INFO,filemode='a')
logging.info("--------------------")
logging.info(f"batch_size={args.batch_size}")
logging.info(f"cluster={k}")
step=0
for c in np.arange(0.0001, 0.1, 0.00198):
    clf = LinearSVC(random_state=0, C=c)
    clf.fit(train_hist, train_label)
    predict = clf.predict(test_hist)
    print ("C =", c, ",\t Accuracy:", np.mean(predict == test_label)*100, "%")
    logging.info(f"step = {step}, Mean accuracy: {np.mean(predict == test_label)*100:.2f}")
    step+=1

#---------------------task 3 visualization: giving confusion matrix to predict results--------------------------#

import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix
cnf_matrix = confusion_matrix(np.array([test_label]).T, predict)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure(figsize=(6, 6))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig("without_normalization.jpg",bbox_inches='tight')
# Plot normalized confusion matrix
plt.figure(figsize=(6, 6))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig("confusion_metrix.jpg",bbox_inches='tight')

import math

def extract_denseSIFT(img):
    DSIFT_STEP_SIZE = 2
    sift = cv2.SIFT_create()
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
            for y in range(0, img.shape[0], disft_step_size)
                for x in range(0, img.shape[1], disft_step_size)]

    descriptors = sift.compute(img, keypoints)[1]
    
    #keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


# form histogram with Spatial Pyramid Matching upto level L with codebook kmeans and k codewords
def getImageFeaturesSPM(L, img, kmeans, k):
    W = img.shape[1]
    H = img.shape[0]   
    h = []
    for l in range(L+1):
        w_step = math.floor(W/(2**l))
        h_step = math.floor(H/(2**l))
        x, y = 0, 0
        for i in range(1,2**l + 1):
            x = 0
            for j in range(1, 2**l + 1):                
                desc = extract_denseSIFT(img[y:y+h_step, x:x+w_step])                
                #print("type:",desc is None, "x:",x,"y:",y, "desc_size:",desc is None)
                predict = kmeans.predict(desc)
                histo = np.bincount(predict, minlength=k).reshape(1,-1).ravel()
                weight = 2**(l-L)
                h.append(weight*histo)
                x = x + w_step
            y = y + h_step
            
    hist = np.array(h).ravel()
    # normalize hist
    dev = np.std(hist)
    hist -= np.mean(hist)
    hist /= dev
    return hist


# get histogram representation for training/testing data
def getHistogramSPM(L, data, kmeans, k):    
    x = []
    for i in range(len(data)):        
        hist = getImageFeaturesSPM(L, data[i], kmeans, k)        
        x.append(hist)
    return np.array(x)
train_histo = getHistogramSPM(2, train_data, kmeans, k)
test_histo = getHistogramSPM(2, test_data, kmeans, k)


# train SVM
for c in np.arange(0.000307, 0.001, 0.0000462):
    clf = LinearSVC(random_state=0, C=c)
    clf.fit(train_histo, train_label)
    predict = clf.predict(test_histo)
    print ("C =", c, ",\t\t Accuracy:", np.mean(predict == test_label)*100, "%")