import os
import shutil
import pandas as pd


# import annotator
from poly2pascal.annotations import XMLAnnotator
import xml.etree.ElementTree as ET

from sklearn.model_selection import train_test_split


# NOTE: using this file under '/yolov5' and create 'test', 'train', 'val' under the '/yolov5/worm_dataset/images'

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"pbw":0,"abw":1}


def load_and_save(src):
  # exmaple src path = '../dataset/'
  # read train data
  train_ids = pd.read_csv(src+'Train.csv')['image_id_worm'].unique().tolist()
  test_ids = pd.read_csv(src+'Test.csv')['image_id_worm'].tolist()

  print(len(train_ids), len(test_ids))

  os.chdir('.../yolov5/worm_dataset')

  # create annotator
  xmla = XMLAnnotator(
      images_path='images',
      csv_file_path='images_bboxes.csv', 
      image_name_col="image_id",
      image_label_col="worm_type", 
      xml_output_path='labels',
      geometry_col="geometry",
      xml_end_content="\n</annotation>"
  )

  # create xml annotation files in Pascal VOC format
  existed = os.listdir('labels')
  for i in train_ids:
    if i[:-3]+"xml" in existed:
      continue
    else:
      xmla.get_all_xml_annotations(img_format=i)


# code sourced from https://blog.paperspace.com/train-yolov5-custom-data/
# credit by Ayoosh Kathuria

# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(float(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = float(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict



# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
    # Name of the file which we have to save 
    save_file_name = os.path.join("labels/", info_dict["filename"].replace("jpg", "txt"))

    # write empty info in the txt if there is no bounding box
    if len(info_dict["bboxes"])==0:
      print("", file=open(save_file_name,"w"))
      return

    print_buffer = []
    # For each bounding box
    for b in info_dict["bboxes"]:
        if b['class'] == 'nan':
          continue
        class_id = class_name_to_id_mapping[b["class"]]
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalize the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))


# copy train and val files to the corresponding folders
def copy_files(src,dest,data):
  for i in data:
    shutil.copy(src+i,dest)

def clean_and_create(srcdir,dir,data):
  # remove all contents in dir
  if os.path.exists(dir):
    for f in os.listdir(dir):
      os.remove(os.path.join(dir,f))
  
  copy_files(srcdir,dir,data)



def prepare_dataset():
  load_and_save('../dataset')

  # connvert XML to txt
  annot = os.listdir('worm_dataset/labels/')
  train_ids = pd.read_csv('../dataset/Train.csv')['image_id_worm'].unique().tolist()

  for i in train_ids:
    if (i[:-3]+'txt') in annot:
      continue
    else:
      convert_to_yolov5(extract_info_from_xml('labels/{}'.format(i)))

  # annotations in txt format
  annot_txt = os.listdir('worm_dataset/labels/')
  annots = [a for a in annot_txt if a[-3:]=='txt']

  # extract the imega ids from annotation list
  # images = pd.Series(train_ids).unique().tolist()
  images = pd.Series(annots).apply(lambda x: x[:-3]+'jpg').tolist()

  # Split the dataset into train-valid-test splits
  annots.sort(), images.sort()
  train_images, val_images, train_annot, val_annot = train_test_split(images[:5000], annots[:5000], 
                                                                      test_size = 0.3, random_state = 1)
  
  # manually create these folder, need to use command to create
  clean_and_create('worm_dataset/images/','worm_dataset/images/train',train_images)
  clean_and_create('worm_dataset/images/','worm_dataset/images/val',val_images)
  clean_and_create('worm_dataset/labels/','worm_dataset/labels/train',train_annot)
  clean_and_create('worm_dataset/labels/','worm_dataset/labels/val',val_annot)