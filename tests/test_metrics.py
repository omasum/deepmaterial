
import os
from deepmaterial.utils import Metrics
import time

if __name__ =='__main__':
    starttime= time.time()
    result_root='results'
    surffix = 'visualization/areaDataset'
    metrics = Metrics("RMSE")
    explist = ['pointNaf45L1Loss_2']
    for exp in explist:
        metrics.svbrdfs_from_dir(os.path.join(result_root, exp, surffix),os.path.join(result_root, exp, surffix), exp_name=exp)
    print("cost time: %.2f seconds"%(time.time()-starttime))
    
# /home/sda/xiaojiu/dataset/testBlended
# /home/sda/svBRDFs/testBlended