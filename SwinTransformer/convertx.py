import os
from PIL import Image

data_dir = '/home/love/Project/dataset/MIA-COV19-DATA/val/covid/'
save_dir = '/home/love/Project/dataset/MIA-COV19-DATA/new/val/covid/'
cases = os.listdir(data_dir)

for case in cases:
    files = os.listdir(data_dir+case)
    for i in files:
        f = os.path.join(data_dir+case, i)
        savef = os.path.join(save_dir, case+'_'+i)
        img = Image.open(f)
        im = img.convert("RGB") #"L" for gray / "RGB" for 3 chainge
        im.save(savef)
        #im = img.covert('L')