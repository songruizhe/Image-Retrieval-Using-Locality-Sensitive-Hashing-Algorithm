#!/usr/bin/env python
# coding: utf-8

# # LSH Algorithm Improvement By Applying Bitmap Indexing

# In[1]:
import argparse
import sys
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Optional, Tuple
import imagehash
from PIL import Image
import os, os.path
import cv2
from collections import Counter
import scipy as sp
import numpy as np # Import numpy library 
from skimage.feature import hog # Import Hog model to extract features
from sklearn.metrics import confusion_matrix # Import confusion matrix to evaluate the performance
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.conf import SparkConf
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split


# In[2]:


imgs = []
y = []
file_size = []
k = 0
path = "./data/101_ObjectCategories" # Give the dataset path here


# ##  Data Preprocessing:
# 1. Load the images using cv2
# 2. Image resize
# 3. Feature extraction: BGR to Gray conversion 
# 4. Feature extraction: Histogram of Oriented Gradients(HOG)

# In[3]:


folder = os.listdir(path) # from the given path get the file names such as accordion, airplanes etc..
for file in folder: # for every file name in the given path go inseide that directory and get the images
    subpath = os.path.join(path,file)  # Join the name of these files to the previous path 
    
    files = os.listdir(subpath) # Take these image names to a list called files
    j = 0
    for i in range(np.size(files)): # now we shall loop through these number of files
        
        im = cv2.imread(subpath+'/'+files[0+j]) # Read the images from this subpath
        
        imgs.append(im) # append all the read images to a list called imgs
        y.append(k) # generate a labe to every file and append it to labels list

        j += 1
        if (j == (np.size(files))):
            file_size.append(j)
   
    k += 1
     
y = np.array(y).tolist()
ix = []
for index, item in enumerate(imgs):
    if (np.size(item) == 1):
        ix.append(index)
        del imgs[index]
        
for index, item in enumerate(y):
    for v in range(np.size(ix)):
        if (index == ix[v]):
            del y[index]
        
y = np.array(y).astype(np.float64) 

# Function to convert an image from color to grayscale
def resize_(image):
    u = cv2.resize(image,(256,256))
    return u

def rgb2gray(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return gray

def fd_hog(image):
    fd = hog(image, orientations=8, pixels_per_cell=(64, 64),
                        cells_per_block=(2, 2))
    
    return fd


# In[4]:


a=[]
import progressbar
with progressbar.ProgressBar(max_value=len(imgs)) as bar:
    i=1
    for img in imgs:
        b=resize_(img)
        c=rgb2gray(b)   
        d=fd_hog(c)
        a.append(d)
        bar.update(i)
        i+=1
df = pd.DataFrame(a)
df['lable'] = y
id_ = np.arange(1,len(df)+1,1)
df['id'] = id_
X = df.values

spark = SparkSession.builder  .master("local")  .appName("Image Retrieval")  .config("spark.some.config.option", "some-value")  .getOrCreate()


# In[5]:


print("HOG diamension: ")
len(a[0])


# In[6]:


# def getBestPerformance(numOfTest, bucketLength,numHashTables,numOfNeighbor):
 
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#     Train = map(lambda x: (int(x[-1]),int(x[-2]),Vectors.dense(x[:-2])), X_train)
#     Train_df = spark.createDataFrame(Train,schema=['id','label',"features"])
#     Test = map(lambda x: (int(x[-1]),int(x[-2]),Vectors.dense(x[:-2])), X_test)
#     Test_df = spark.createDataFrame(Test,schema=['id','label',"features"])

#     brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", 
#                                       bucketLength=bucketLength,numHashTables=numHashTables)
    
    
#     model = brp.fit(Train_df)
#     model.transform(Train_df)
  
#     nnToAccMap = {}
#     with progressbar.ProgressBar(max_value = numOfTest) as bar:
#         for i in range(0, numOfTest):
#             Catg = X_test[i][-2]
#             key = Vectors.dense(X_test[i][0:-2])
#             # Choose the Last one of numOfNeighbor, the biggest one 
#             result = model.approxNearestNeighbors(Train_df, key, numOfNeighbor[-1])
#             # Conver pySpark framework colunm to python list
#             labelList = result.select("label").rdd.flatMap(lambda x: x).collect()
#             # slice LableList into differnt length subLists 
#             nnList = []
#             for numberNN in numOfNeighbor:
#                 slicedList = labelList[0:numberNN]
#                 nnList.append(slicedList)
                
#             for index in range(0, len(nnList)):
#                 majority_vote = Counter(nnList[index]).most_common(1)[0][0]
#                 if  Catg == majority_vote:
#                     key = numOfNeighbor[index]
#                     if key in nnToAccMap:
#                         nnToAccMap[key] = nnToAccMap.get(key) + 1
#                     else:
#                         nnToAccMap[key] = 1
#             bar.update(i)
#         # calucate accuracy
#         for key in nnToAccMap:
#             nnToAccMap[key] = nnToAccMap.get(key) / numOfTest
#     return nnToAccMap


# In[11]:


# #set Param 
# bucketLengthList = np.arange(0, 50, 10)
# numHashTablesList = np.arange(0, 150, 30)
# bucketLengthList[0] = 1
# numHashTablesList[0] = 1
# #make sure the last element of numOfNeighborList is the largest
# numOfNeighborList = [1, 3, 5, 7, 15, 21, 25];
# numOfTest = 5
# print("Checking bucketLength Param:")
# print(bucketLengthList)
# print("Checking numHashTablesList Param:")
# print(numHashTablesList)


# In[12]:


# %%time
# bucketLengthList_para=[]
# numHashTablesList_para=[]
# resultList = []
# for i in bucketLengthList:
#     for j in numHashTablesList:
#             result = getBestPerformance(numOfTest ,i, j, numOfNeighborList)
#             print( "bucketLen:" + str(i) + "  #Hashtable:" + str(j) +  " Acc: " + str(result))
#             bucketLengthList_para.append(i)
#             numHashTablesList_para.append(j)
#             resultList.append(result)


# In[13]:


# df_result = pd.DataFrame()
# df_result['BucketLength'] = bucketLengthList_para
# df_result['NumHashTables'] = numHashTablesList_para
# # conver List of map to panda dataframework
# df_result_map = pd.DataFrame(resultList)
# df_result = pd.concat([df_result, df_result_map], axis=1, join='inner')

# # df_result["Acc"] = resultList
# # df_result = df_result.sort_values(by=['Acc'],ascending=False)
# df_result.to_csv('./result2.csv') #Chang the name every you wanna sava a file


# In[10]:


# df_result


# In[ ]:





# # !!!Skip all the code below !!!

# ## Split data
# Split the data to training and validation data. We choose 70% for training and 30% for validation purposes.

# In[14]:


df = pd.DataFrame(a)
df['lable'] = y
id_ = np.arange(1,len(df)+1,1)
df['id'] = id_
X = df.values

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ## Using PySpark to retrieve similar images

# In[ ]:





# In[16]:


spark = SparkSession.builder      .master("local")      .appName("Image Retrieval")      .config("spark.some.config.option", "some-value")      .getOrCreate()


# In[17]:



Train = map(lambda x: (int(x[-1]),int(x[-2]),Vectors.dense(x[:-2])), X_train)
Train_df = spark.createDataFrame(Train,schema=['id','label',"features"])


# In[18]:


Test = map(lambda x: (int(x[-1]),int(x[-2]),Vectors.dense(x[:-2])), X_test)
Test_df = spark.createDataFrame(Test,schema=['id','label',"features"])


# In[19]:


Train_df.show(n = 2)


# # !!!!! Skip以下代码直接运行最后一行 !!!!!

# In[20]:


## skip以下代码直接运行最后一行


# In[30]:



brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes",bucketLength=2,numHashTables=3)
model = brp.fit(Train_df)
print("The hashed dataset where hashed values are stored in the column 'hashes':")
model.transform(Train_df).show()
 
# In[31]:


key = Vectors.dense(X_test[0][0:-2])


# In[32]:


key


# In[33]:


X_test[0][-2]


# In[34]:


print("Approximately searching Train_df for 2 nearest neighbors of the key:")
@profile
def measure(Train_df, key) :
    model.approxNearestNeighbors(Train_df, key, 5).show()
    return model
model = measure(Train_df, key)

# In[35]:


# result_id = result.select('label',).collect()
# result_id[0].label


# In[36]:


# print("Approximately joining Train_df and Test_df on Euclidean distance smaller than 1:")
# model.approxSimilarityJoin(Train_df, Test_df, 1.1, distCol="EuclideanDistance")\
#     .select(col("datasetA.id").alias("Train_df"),
#             col("datasetB.id").alias("Test_df"),
#             col("EuclideanDistance")).show(30)


# In[37]:


accuracy = 0
numOfNeighbor = 5
numOfTest= 5
accList = []
with progressbar.ProgressBar(max_value=numOfTest) as bar:
    for i in range(0, numOfTest):
        Catg = X_test[i][-2]
        key = Vectors.dense(X_test[i][0:-2])
        result = model.approxNearestNeighbors(Train_df, key, numOfNeighbor)
        temp = Counter([int(row['label']) for row in result.collect()])
        if  Catg in temp:
            accuracy += temp.get(Catg)/ numOfNeighbor
            accList.append(temp.get(Catg)/ numOfNeighbor)
        else:
            accList.append(0)
        bar.update(i)
    accuracy /= numOfTest


# In[38]:


accuracy


# In[39]:


print(accList)


# In[ ]:


# from matplotlib.pyplot import imshow
# imshow(imgs[4795])


# In[ ]:




