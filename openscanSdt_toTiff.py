#run in base env
import os
import glob
import tifffile as tf
from sdt_read.read_openscan_sdt import *

os.chdir(r'E:\FLIM_Data\2025_7_8\KO\740\Data\Sample_1\LOC_1')
filelist = glob.glob(r'*.sdt')
print(filelist)

for filename in filelist:
    data,_,_ = read_sdt_openscan(filename)
    tf.imwrite(filename[:-4]+'_tf.tif',data[0].transpose(2,0,1).astype(np.uint16))
    print(filename, data[0].shape)
    
print("done")


import numpy as np
import tifffile as tf

for filename in filelist:
    data, _, _ = read_sdt_openscan(filename)
    img2d = data[0].sum(axis=2)  # Sum across time bins

    # Stretch intensities for visibility
    img2d = img2d.astype(np.float32)
    img2d -= img2d.min()
    if img2d.max() > 0:
        img2d /= img2d.max()
        img2d *= 65535
    img2d = img2d.astype(np.uint16)

    tf.imwrite(filename[:-4] + '_intensity_scaled.tif', img2d)
    print(f"{filename} saved: shape={img2d.shape}, max={img2d.max()}")

print("done")

