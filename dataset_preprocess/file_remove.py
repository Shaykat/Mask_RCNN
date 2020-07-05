import os


def file_remove(path):
    base_name = {}
    for f in os.listdir(path):
        # if f.endswith('.xml') or f.endswith('.jpg'):
        base = f[:-4]
        if base in base_name and base_name[base]:
            base_name[base] += 1
        else:
            base_name.update({base: 1})

    for k, v in base_name.items():
        # if v == 1:
        file_name = os.path.join(path, k+'.xml')
        if os.path.isfile(file_name):
            os.remove(file_name)

path = "/home/jupyter/dataset/val/"
file_remove(path)
