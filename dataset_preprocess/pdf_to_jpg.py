from pdf2image import convert_from_path, convert_from_bytes
import PyPDF2
import os
from PIL import Image
import PIL

path = '/home/jupyter/dataset_/CSC_PPR/SPLITTED_CSC_PPR_ver2/'
output_path = '/home/jupyter/dataset_/CSC_PPR/JPG_CSC_PPR_ver2/'
cnt = 0
for filename in os.listdir(path):
    if filename.endswith(".pdf"):
        try:
            images = convert_from_path(os.path.join(path, filename), 50)
        except ValueError:
            print(filename + " is not converting to jpg")
        image_path=os.path.join(output_path, filename[:-3]+"jpg")
        if not os.path.isfile(image_path):
            print(image_path)
            cnt = cnt + 1
            print(cnt)
            images[0].save(os.path.join(output_path, filename[:-3]+"jpg"))