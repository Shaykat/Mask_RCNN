import sys
import os
import csv


def str_to_int(val):
    if val.isdigit():
        return val
    else:
        p = ""
        for v in val:
            if v.isdigit():
                p += v
            if v == '.':
                p += v
        return p


def csv_to_xml(path, path_in):
    print(path)
    csvFiles = [f for f in os.listdir(path_in) if f.endswith('.csv') or f.endswith('.CSV')]
    print(len(csvFiles))
    for csvFile in csvFiles:
        print(csvFile)
        csvData = csv.reader(open(os.path.join(path_in, csvFile)), delimiter=",")
        file_name = "filename"
        rowNum = 0
        for row in csvData:
            if rowNum == 0:
                rowNum += 1
                continue
            xmlFile = str(row[0])[:-3] + 'xml'
            xmlFile = os.path.join(path, xmlFile)

            xmlData = open(xmlFile, 'w')
            xmlData.write('<annotation>' + "\n")
            xmlData.write('    <folder>Images</folder>' + "\n")

            for i in range(len(row)):
                if i == 0:
                    xmlData.write('    <' + file_name + '>' + row[i] + '</' + file_name + '> \n')
                elif i == 1 and len(row) > 2:
                    xmlData.write('    <size>' + "\n")
                    xmlData.write('        <width>' + row[1] + '</width>\n')
                    xmlData.write('        <height>' + row[len(row)-1] + '</height>\n')
                    xmlData.write('        <depth>' + "1" + '</depth>\n')
                    xmlData.write('    </size>' + "\n")
                elif i == 2:
                    labels = row[i].split(',')
                    for j in range(len(labels)):
                        if (j+1) % 4 == 0:
                            xmlData.write('    <object>' + "\n")
                            xmlData.write('        <bndbox>' + '\n')
                            xmlData.write('            <xmin>' + str_to_int(labels[j-3]) + '</xmin>\n')
                            xmlData.write('            <ymin>' + str_to_int(labels[j-2]) + '</ymin>\n')
                            xmlData.write('            <xmax>' + str_to_int(labels[j-1]) + '</xmax>\n')
                            xmlData.write('            <ymax>' + str_to_int(labels[j]) + '</ymax>\n')
                            xmlData.write('        </bndbox>' + '\n')
                            xmlData.write('    </object>' + '\n')

                # xmlData.write('    ' + '<' + tags[i] + '>' + row[i] + '</' + tags[i] + '>' + "\n")
            print(rowNum)
            rowNum += 1
            xmlData.write('</annotation>' + "\n")
            xmlData.close()


path = "/home/jupyter/pod/dataset_nik/"
path_in = "/home/jupyter/pod/Mask_RCNN/"
csv_to_xml(path, path_in)

