#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random


# In[2]:


trainval_percent = 0.8
train_percent = 0.75
xmlfilepath = r"D:/22Fall/EE641.Deep Learning Systems/project/mmdetection/data/VOCdevkit/VOC2007/Annotations"
txtsavepath = r"D:/22Fall/EE641.Deep Learning Systems/project/mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main"


# In[3]:


random.seed(4)


# In[4]:


total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)


# In[5]:


ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')


# In[6]:


for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)


# In[7]:


ftrainval.close()
ftrain.close()
fval.close()
ftest.close()


# In[ ]:




