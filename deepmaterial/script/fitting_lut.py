from asyncio.log import logger
from deepmaterial.script.fitting import *
import sys
import time

class Logger(object):
    '''Printing the log information to the file

    Args:
        fileN (path): the path of log file
    '''
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(fileN, 'w')

    def write(self, message):
        '''logger.write实际相当于sys.stdout.write'''
        self.terminal.write(message)
        self.log.write(message)

    def reset(self):
        self.log.close()
        sys.stdout=self.terminal
    
    def flush(self):
        self.log.flush()

def fitting_lut(logger, nSampleR=64, nSampleV=256, saveImgs=False, outDir='tmp'):
    '''Fitting the look-up-table of brdf

    Args:
        logger (path): the path of log file
        nSample (int, optional): the resolution of the look-up-table. Defaults to 64.
        saveImgs (bool, optional): whether save images during fitting. Defaults to True.
        outDir (path, optional): the folder to save images. Defaults to 'tmp'.
    '''
    MIN_ROUGHNESS=np.sqrt(0.001)
    lut = torch.zeros((nSampleR, nSampleV, 6), dtype=torch.float32)
    for i in range(nSampleR):
        a = nSampleR - 1 - i
        for t in range(nSampleV):
            #* initialize parameters of the look-up-table for fitting
            theta = np.minimum(89.95, t / (nSampleV-1) * 90.0)
            roughness = np.maximum(MIN_ROUGHNESS, a / (nSampleR-1))
            
            logger.write("a = %d\t t = %d\n"%(a, t))
            logger.write("alpha = %.2f\t theta = %.2f\n"%(roughness**2, theta))

            #* set the initialization of optimization for better converge
            if t == 0:
                Z = [0,0,1]
                if a == nSampleR - 1:
                    initialGuess = [1.0, 1.0, 0.0, 0.0]
                else:
                    #* search the result for the initialization of next iteration
                    initialGuess = lut[a+1, t, :4]
            else:
                Z = None
                initialGuess = preGuess
            
            #* fitting
            res, ltc, brdf, fresNorm = fit(theta, roughness * 2 - 1, Z=Z, initialGuess=initialGuess)
            logger.write("final error: %.6f\n" % res.final_simplex[1][0])
            logger.write(res.message+'\n')
            logger.write('result: ' + str(res.x)+'\n')
            
            #* normalize the transform matrix using m22
            m22 = ltc.M[2,2]
            #* saving the transform matrix and normalization factors
            lut[a, t] = torch.from_numpy(np.array([ltc.M[0,0]/m22, ltc.M[1,1]/m22, ltc.M[0,2]/m22, ltc.M[2,0]/m22, ltc.amplitude, fresNorm]))

            #* store the result for the initialization if next iteration
            preGuess = res.x

            logger.write('=======================================\n')
            logger.flush()
            #* plot the fitting result
            if saveImgs:
                plotResult(ltc, brdf, roughness=roughness * 2 - 1, theta=theta, nSample=128, name=osp.join(outDir, 'res_a%2d_t%2d'%(a, t)))
    torch.save(lut, 'deepmaterial/utils/LTC/look-up-table-large.pth')

if __name__=="__main__":
    logger = Logger('./tmp/fitting.log')
    fitting_lut(logger = logger, outDir='tmp1')
    logger.reset()
