import urllib
import csv
import os.path as pth
import os
import xml.etree.ElementTree as ET


files = []
dataset_dir = "/home/jupyter/dataset/train"
try:
    for filename in os.listdir(dataset_dir):
        if not filename.endswith('.xml'):
            continue

        path = os.path.join(dataset_dir, filename)
        if pth.isfile(path):
            tree = ET.parse(path)
            root = tree.getroot()
            for file_name in root.iter('filename'):
                file_name.text = filename[:-3] + "jpg"
            for width in root.iter('width'):
                width.text = str(round(int(width.text)/2))
            for height in root.iter('height'):
                height.text = str(round(int(height.text)/2))
            for xmin in root.iter('xmin'):
                xmin.text = str(round(int(xmin.text)/2))
            for ymin in root.iter('ymin'):
                ymin.text = str(round(int(ymin.text)/2))
            for xmax in root.iter('xmax'):
                xmax.text = str(round(int(xmax.text)/2))
            for ymax in root.iter('ymax'):
                ymax.text = str(round(int(ymax.text)/2))
            print(root)
            tree.write(path)
except IOError:
    print("File 'annotations.csv' is no exist")

