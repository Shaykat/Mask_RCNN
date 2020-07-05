import pandas as pd
import os

dataset_dir = "/refined/funsd/train/Annotation/"

for filename in os.listdir('.'):
    if not filename.endswith('.json'):
        continue
    df = pd.read_json('82092117.json')['form']

    file_name = "filename"
    xmlFile = filename[:-4] + 'xml'
    imgFile = filename[:-4] + 'png'


    xmlData = open(xmlFile, 'w')
    xmlData.write('<annotation>' + "\n")

    xmlData.write('    <' + file_name + '>' + imgFile + '</' + file_name + '> \n')
    xmlData.write('    <size>' + "\n")
    xmlData.write('        <width>' + '754' + '</width>\n')
    xmlData.write('        <height>' + '1000' + '</height>\n')
    xmlData.write('        <depth>' + "1" + '</depth>\n')
    xmlData.write('    </size>' + "\n")
    xmlData.write('    <object>' + "\n")

    for key, value in df.items():
        box = value['box']
        xmin = box[0]
        xmax = box[2]
        ymin = box[1]
        ymax = box[3]

        xmlData.write('        <bndbox>' + '\n')
        xmlData.write('            <xmin>' + str(xmin) + '</xmin>\n')
        xmlData.write('            <ymin>' + str(ymin) + '</ymin>\n')
        xmlData.write('            <xmax>' + str(xmax) + '</xmax>\n')
        xmlData.write('            <ymax>' + str(ymax) + '</ymax>\n')
        xmlData.write('        </bndbox>' + '\n')
    xmlData.write('    </object>' + '\n')
    xmlData.write('</annotation>' + "\n")
    xmlData.close()
