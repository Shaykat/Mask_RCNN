import os
import glob


def file_move_remove(path_from, path_to):
    base_name = {}
    for f in os.listdir(path_from):
        # if f.endswith('.xml') or f.endswith('.jpg'):
        base = f[:-4]
        if base in base_name and base_name[base]:
            base_name[base] += 1
        else:
            base_name.update({base: 1})
    dic_len = int(len(base_name)/3)

    for i, (k, v) in enumerate(base_name.items()):
        if v == 2:
            if i < dic_len:
                os.rename(os.path.join(path_from, k +'.jpg'), os.path.join(path_to, k +'.jpg'))
                os.rename(os.path.join(path_from, k +'.xml'), os.path.join(path_to, k +'.xml'))
        else:
            file_names = glob.glob(os.path.join(path_from, k+'.*'))
            # if os.path.isfile(file_name):
            for file_name in file_names:
                try:
                    os.remove(file_name)
                except:
                    print("Error while deleting file : ", file_name)
            
            


path_from = "/home/jupyter/refined/dataset/train/"
path_to = "/home/jupyter/refined/dataset/val/"
file_move_remove(path_from, path_to)

