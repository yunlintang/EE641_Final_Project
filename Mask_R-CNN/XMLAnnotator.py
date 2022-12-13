#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install poly2pascal


# In[4]:


# import annotator
from poly2pascal.annotations import XMLAnnotator

# create annotator
xmla = XMLAnnotator(
    images_path="D:/22Fall/EE641.Deep Learning Systems/project/mmdetection/data/VOCdevkit/VOC2007/JPEGImages", 
    csv_file_path="D:/22Fall/EE641.Deep Learning Systems/project/images_bboxes.csv", 
    image_name_col="image_id",
    image_label_col="worm_type", 
    xml_output_path="D:/22Fall/EE641.Deep Learning Systems/project/mmdetection/data/VOCdevkit/VOC2007/Annotations",
    geometry_col="geometry",
    xml_end_content="\n</annotation>"
)

# create xml annotation files in Pascal VOC format
xmla.get_all_xml_annotations(img_format=".jpg")


# In[ ]:




