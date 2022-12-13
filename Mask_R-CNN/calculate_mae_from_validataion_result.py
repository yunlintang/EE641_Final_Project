#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas


# In[2]:


# read result.pkl
img_path = './result/result.pkl'
img_data = np.load(img_path, allow_pickle=True)
print(len(img_data))
print(img_data)


# In[3]:


# check prediction
#i = 5
#print(img_data[i])
#print(len(img_data[i][0]), len(img_data[i][1])) # pbw, abw


# In[4]:


# record the prediction which has possibility higher than the threshold
threshold = 0.3

pred_num = []
for i in range(len(img_data)):
    pred = [0, 0] # [pbw, abw]
    for j in [0, 1]:
        for info in img_data[i][j]:
            if info[-1] >= threshold:
                pred[j] += 1
    pred.reverse() # [abw, pbw]
    pred_num = pred_num + pred
print(len(pred_num))


# In[5]:


# read Train.csv
all_file = pandas.read_csv("../Train.csv")
print(all_file)


# In[6]:


# read test.txt file (data with true num from Train.csv)
test_file_path = './data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
with open(test_file_path) as f:
    test_file = f.readlines()
for i in range(len(test_file)): #remove '\n' from every row
    if test_file[i][-1:] == '\n':
        test_file[i] = test_file[i][:-1]


# In[7]:


#print(test_file)
print(len(test_file))


# In[8]:


# get true num from the Train.csv with corresponding image id in test.txt
true_num = []

for i in range(len(test_file)):
    abw_pbw = [0, 0]
    find_test_jpg = all_file.loc[all_file['image_id_worm'] == test_file[i]+'.jpg'] # image file could be .jpg or .jpeg
    find_test_jpeg = all_file.loc[all_file['image_id_worm'] == test_file[i]+'.jpeg']
    if len(find_test_jpg) > 0:
        find_test = find_test_jpg.values
    else:
        find_test = find_test_jpeg.values
    #print(find_test)

    for j in range(len(find_test)):
        if find_test[j][1] == 'pbw':
            abw_pbw[1] = find_test[j][2]
        elif find_test[j][1] == 'abw':
            abw_pbw[0] = find_test[j][2]
    true_num = true_num + abw_pbw
print(len(true_num))


# In[9]:


# calculate mean absolute error
mae = 0
for i in range(len(true_num)):
    mae += abs(pred_num[i] - true_num[i])
mae /= len(true_num)/2
print(mae)

