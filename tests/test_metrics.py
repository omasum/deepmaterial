
import os
from deepmaterial.utils import Metrics
import time

if __name__ =='__main__':
    starttime= time.time()
    result_root='/home/sda/cjm/Results'
    surffix = 'visualization/areaDataset'
    metrics = Metrics("RMSE")
    explist = ['0016_MSAN']
    for exp in explist:
        metrics.svbrdfs_from_dir("/home/sda/svBRDFs/testBlended",os.path.join(result_root, exp, surffix), exp_name=exp)
    print("cost time: %.2f seconds"%(time.time()-starttime))
    
# /home/sda/xiaojiu/dataset/testBlended
# /home/sda/svBRDFs/testBlended