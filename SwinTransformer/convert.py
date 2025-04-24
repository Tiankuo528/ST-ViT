import os
from PIL import Image
import json
from os.path import join,dirname,exists,basename
import SimpleITK as sitk
import numpy as np
data_dir = '/home/love/Project/dataset/MIA-COV19-DATA/'
save_dir = '/home/love/Project/dataset/MIA-COV19-DATA/newx/'
bbox_mask_dir ='.'
# data_dir = '/media/yanwe/comp/Data/Lung/3D/ICCV/MIA-COV19-DATA/'
# save_dir = '/media/yanwe/comp/Data/Lung/3D/ICCV/MIA-COV19-DATA/newx/'
# bbox_mask_dir = '/media/yanwe/新加卷/Data/Lung/ICCV_Lung/'
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_bbox(np_mask):
    objs_amount = 2
    locs = ndimage.find_objects(np_mask)
    if isinstance(locs, list):
        if not len(locs) == objs_amount:
            raise Exception
    bbox = [[min([loc[xyz].start for loc in locs]),
             max([loc[xyz].stop for loc in locs])]
            for xyz in range(0, 3)]
    return bbox

def process_case(case):
    maskf = join(bbox_mask_dir, case['path'], 'mask.nii.gz')
    mask = sitk.ReadImage(maskf)
    np_mask = sitk.GetArrayFromImage(mask)
    np_mask = np.flip(np_mask, axis=0)
    bbox=get_bbox(np_mask)
    zmin = bbox[0][0]
    zmax = bbox[0][1]
    ymin = bbox[1][0]
    ymax = bbox[1][1]
    xmin = bbox[2][0]
    xmax = bbox[2][1]
    if not exists(join(save_dir, dirname(case['path']))):
        os.makedirs(join(save_dir, dirname(case['path'])))
    for i in range(zmin, zmax):
        f = join(data_dir, case['path'], str(i) + '.jpg')
        savef = join(save_dir, case['path'] + '_' + str(i) + '.jpg')
        slice_mask = np.clip(np_mask[i], 0, 1)
        np_image = np.array(Image.open(f))
        np_image = np_image * slice_mask
        np_image = np_image[ymin:ymax, xmin:xmax]
        np_image = np.stack([np_image, np_image, np_image], axis=2)
        image = Image.fromarray(np_image)
        image.save(savef)
    print(join(save_dir,case['path']))

with open(join(bbox_mask_dir,'bbox.json')) as json_file:
    cases = json.load(json_file)
    with ThreadPoolExecutor(max_workers=36) as executor:
        for case in cases:
            executor.submit(process_case,case)



# cases = os.listdir(data_dir)
#
# for case in cases:
#     files = os.listdir(data_dir+case)
#     for i in files:
#         f = os.path.join(data_dir+case, i)
#         savef = os.path.join(save_dir, case+'_'+i)
#         img = Image.open(f)
#         im = img.convert("RGB") #"L" for gray / "RGB" for 3 chainge
#         im.save(savef)
#         #im = img.covert('L')