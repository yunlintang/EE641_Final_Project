#!/usr/bin/env python
# coding: utf-8

# In[13]:


import json
import numpy as np
import matplotlib.pyplot as plt


# In[36]:


file_path = './work_dirs/mask_rcnn_r50_fpn_1x_voc_try002/20221209_231304.log.json'
#file_path = './work_dirs/mask_rcnn_r101_fpn_1x_voc_try001/20221209_031758.log.json'

data = [json.loads(line) for line in open(file_path, 'r')]    # one object per line


# In[37]:


print(type(data))
print(len(data))
print(type(data[0]))
print(data[1]['acc'])


# In[38]:


#    calculate average accuracy in each epoch
acc = [] # average train accuracy (len = epoch)
acc_in_epoch = []
epoch = data[-1]['epoch'] # total epoch trained
e = 1 # start from the first epoch
mAP = [] # mean averege precise

for i in range(1, len(data)):
    if data[i]['mode'] == 'train':
        acc_in_epoch.append(data[i]['acc'])
    else: # mode = val
        acc.append(np.average(acc_in_epoch))
        acc_in_epoch = []
        e += 1
        
        mAP.append(data[i]['mAP'])


# In[43]:


#    plot accuracy curve
plt.plot(range(1, epoch+1), acc, marker='o')
plt.xticks(range(1, epoch+1))
plt.xlabel('epoch'), plt.ylabel('accuracy'), plt.title('Training Accuracy'), plt.grid()
plt.show()


# In[44]:


#    plot mAP curve
plt.plot(range(1, epoch+1), mAP, marker='o')
plt.xticks(range(1, epoch+1))
plt.xlabel('epoch'), plt.ylabel('mAP'), plt.title('Validation Mean Average Precision'), plt.grid()
plt.show()


# In[45]:


# plot recall and ap curve
# pbw, abw
e1_recall = [0.619, 0.962]
e1_ap = [0.536, 0.887]
e2_recall = [0.657, 0.983]
e2_ap = [0.571, 0.902]
e3_recall = [0.646, 0.980]
e3_ap = [0.578, 0.903]
e4_recall = [0.669, 0.978]
e4_ap = [0.597, 0.904]
e5_recall = [0.667, 0.975]
e5_ap = [0.595, 0.905]

e_recall = [np.average(e1_recall), np.average(e2_recall), np.average(e3_recall), np.average(e4_recall), np.average(e5_recall)]
e_ap = [np.average(e1_ap), np.average(e2_ap), np.average(e3_ap), np.average(e4_ap), np.average(e5_ap)]

plt.plot(range(1, epoch+1), e_ap, marker='o', label='precision')
plt.plot(range(1, epoch+1), e_recall, marker='o', label='recall')
plt.xticks(range(1, epoch+1))
plt.xlabel('epoch'), plt.grid(), plt.legend()
plt.show()


# In[ ]:




