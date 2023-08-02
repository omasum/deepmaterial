import os
from deepmaterial.metrics.psnr_ssim import ssim
import time

if __name__ =='__main__':
    starttime= time.time()
    result_root='/home/klkjjhjkhjhg/code/DeepMaterial/results'
    surffix = 'visualization/areaDataset'
    metrics = Metrics("RMSE")
    explist = ['003_LSDN', '006_LSDN']
    for exp in explist:
        metrics.svbrdfs_from_dir("/home/sda/svBRDFs/testBlended",os.path.join(result_root, exp, surffix), exp_name=exp)