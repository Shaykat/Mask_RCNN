import os
import numpy as np
from sklearn.model_selection import train_test_split


def file_remove(path, new_path):
    base_name = {}
    file_list = []
    for f in os.listdir(path):
        # if f.endswith('.xml') or f.endswith('.jpg'):
        base = f[:-4]
        if base in base_name and base_name[base]:
            base_name[base] += 1
            file_list.append(base)
        else:
            base_name.update({base: 1})

    for k, v in base_name.items():
        if v == 1:
            file_name = os.path.join(path, k+'.jpg')
            if os.path.isfile(file_name):
                os.remove(file_name)
    
    train, val = train_test_split(file_list, test_size=0.30, random_state=42)
    
    print(len(train))
    for base in train:
        file_name_jpg = os.path.join(path, base+".jpg")
        file_name_jpg_dest = os.path.join(new_path, base + ".jpg")
        file_name_xml = os.path.join(path, base+".xml")
        file_name_xml_dest = os.path.join(new_path, base + ".xml")
        
        if os.path.isfile(file_name_jpg) and os.path.isfile(file_name_xml):
            os.rename(file_name_jpg, file_name_jpg_dest)
            os.rename(file_name_xml, file_name_xml_dest)
    print(len(val))


path = "/home/jupyter/pod/dataset/train/"
new_path = "/home/jupyter/pod/dataset/val/"
file_remove(path, new_path)