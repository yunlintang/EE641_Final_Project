# EE 641 Final Project: Image Object Detection - Counting Bollworms
Author: Kai-Wen Cheng, Yixiang Zheng, Yunlin Tang

## Set Up
- python3
- If needed, install the used modules included in the requirements.txt:
```
pip install -r requirements.txt
```
- for YOLO model:
  - ```git clone https://github.com/ultralytics/yolov5```
  - ```cd yolov5```
  - ```pip install -r requirements.txt```
  - ```mkdir worm_dataset```
  - ```cd worm_datasert```
  - ```cp -r ../dataset/images /worm_dataset/```
  - ```mkdir labels```

## Repository Structure
```
├── dataset
│   ├── Test.csv
│   ├── Train.csv
│   ├── images.zip
│   ├── images_bboxes.csv
├── FASTERRCNN
│   ├── data.ipynb
│   ├── evaluation.ipynb
├── MASK_R-CNN
│   ├── calculate_mae_from_validataion_result.py
│   ├── mask_rcnn_r101_fpn_1x_voc.py
│   ├── plot_acc_mAP.py
│   ├── separate_data.py
│   ├── XMLAnnotator.py
├── YOLO
│   ├── model_training.txt
│   ├── prepare_data.py
│   ├── worm_data.yaml
│   ├── yolo_notebook.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset
- download dataset from [the ZIND! website](https://zindi.africa/competitions/wadhwani-ai-bollworm-counting-challenge/data) and stored these files in `dataset` folder
- unzip the `images.zip` results in `images` folder

## References
- convert POLYGON bounding boxes to XML: https://github.com/ruankie/poly2pascal
- prepare annotations for YOLO from XML: https://blog.paperspace.com/train-yolov5-custom-data/
- yolov5: https://github.com/ultralytics/yolov5
- mmdetection on VOC: https://blog.csdn.net/AI414010/article/details/109513369
- mmdetection github: https://github.com/open-mmlab/mmdetection


