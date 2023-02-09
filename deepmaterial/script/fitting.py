import torch
import numpy as np
from scipy import optimize
from deepmaterial.utils.ltc_util import LTC
from deepmaterial.utils.render_util import PlanarSVBRDF, RectLighting
import matplotlib.pyplot as plt
from matplotlib import cm
import os.path as osp
import os

from deepmaterial.utils.vector_util import numpy_norm

def computeError(params, ltc: LTC, brdf: PlanarSVBRDF, roughness, viewDir, n, log=False, idx={"value": 1}, nSample=1024):
    '''calculate the error between the evaluation of current ltc and ground truth brdf using importance sampling

    Args:
        params (array): parameters to fitting
        ltc (LTC): current ltc for evaluation and fitting
        brdf (PlanarSVBRDF): ground truth brdf for evaluation and fitting
        roughness (float): roughness of the brdf and ltc
        viewDir (vec): view direction if the brdf and ltc
        n (vec): normal
        log (bool, optional): whether log the error and step. Defaults to False.
        idx (dict, optional): the parameter changed every iteration for log the step. Defaults to {"value": 1}.
        nSample (int, optional): number of sample for fitting. Defaults to 1024.

    Returns:
        error: the current error for the optimization
    '''
    m00, m11, m02, m20 = params
    ltc.update(m00, m11, m02, m20)

    u = torch.from_numpy((np.arange(0, nSample, 1, dtype=np.float32) + 0.5)) / nSample
    v = torch.from_numpy((np.arange(0, nSample, 1, dtype=np.float32) + 0.5)) / nSample
    u, v = torch.meshgrid(u, v, indexing='ij')
    brdf.setPointSet(u,v)

    #* importance sample LTC
    l = ltc.sample(u, v)
    eval_brdf, pdf_brdf, l, h = brdf.eval(n=torch.from_numpy(n), l=l, r=roughness, v=torch.from_numpy(
        viewDir), useDiff=False, useFre=False, importantSampling=True)

    eval_ltc = ltc.eval(l, n)
    pdf_ltc = eval_ltc / ltc.amplitude

    error_ = np.abs(eval_brdf - eval_ltc)
    error = error_**3 / (pdf_ltc + pdf_brdf)
    
    #* importance sample BRDF
    eval_brdf, pdf_brdf, l, h = brdf.eval(n=torch.from_numpy(n), r=roughness, v=torch.from_numpy(
        viewDir), useDiff=False, useFre=False, importantSampling=True)

    eval_ltc = ltc.eval(l, n)
    pdf_ltc = eval_ltc / ltc.amplitude

    error_ = np.abs(eval_brdf - eval_ltc)
    error += error_**3 / (pdf_ltc + pdf_brdf)
    
    idx["value"] += 1
    
    error = error.mean().numpy()
    if log:
        print("step: %4d, error: %.6f" % (idx['value'], error))
    return error

def plotVec(ax, vec):
    vec = np.concatenate([vec, vec[0:1]], axis=0)*2
    for i in range(vec.shape[0]-1):
        ax.plot3D(-vec[i:i+2,0], vec[i:i+2,1], vec[i:i+2,2], color='black')
        ax.plot3D(-np.array([0,vec[i,0]]), np.array([0,vec[i,1]]), np.array([0,vec[i,2]]), color='black')
        norm = numpy_norm(vec[i], dim=0)*2
        ax.scatter3D(-norm[0], norm[1], norm[2], color="black", depthshade=True)

    return ax

def plotResult(ltc: LTC, brdf: PlanarSVBRDF, roughness, theta, rect=None, nSample=256, jointPlot=False, name='result'):  
    '''plot the fitting result to image

    Args:
        ltc (LTC): final ltc
        brdf (PlanarSVBRDF): ground truth brdf
        roughness (float): roughness
        theta (float): angle of the view direction
        nSample (int, optional): resolution of the generated images. Defaults to 256.
        jointPlot (bool, optional): whether joint plot the three images in a figure. Defaults to False.
        name (str, optional): name prefix of the images. Defaults to 'result'.
    '''
    stepu = torch.pi / nSample
    stepv = 2*torch.pi / nSample
    thetaArc = theta/180*torch.pi
    viewDir = np.array([np.sin(thetaArc), 0, np.cos(thetaArc)], dtype=np.float32).reshape(1, 3, 1, 1)

    #* sample the sphrical vectors for evaluation
    u = torch.arange(0, torch.pi, stepu)
    v = torch.arange(0, 2*torch.pi, stepv)
    u, v = torch.meshgrid(u, v, indexing='ij')
    vec = torch.stack([torch.sin(u)*torch.cos(v), torch.sin(u)*torch.sin(v), torch.cos(u)], dim=0).unsqueeze(0)
    n = torch.from_numpy(np.array([0, 0, 1], dtype=np.float32).reshape(1, 3, 1, 1))

    #* evaluation
    evalDo = ltc.DoClampedCos(vec, n)    
    evalLTC = ltc.eval(vec, n)    
    evalBRDF, pdf_brdf, l, h = brdf.eval(n=n, l=vec, r=roughness, v=torch.from_numpy(
        viewDir), useDiff=False, useFre=False, importantSampling=True)
    
    #* normalization for display
    evalDo = evalDo.squeeze(0).squeeze(0) / evalDo.max()
    evalLTC = evalLTC.squeeze(0).squeeze(0) / evalLTC.max()
    evalBRDF = evalBRDF.squeeze(0).squeeze(0) / evalBRDF.max()
    if not jointPlot:
        #* define the name of output images and create output folders
        base, file = osp.split(name)
        os.makedirs(osp.join(base, 'ltc'), exist_ok=True)
        os.makedirs(osp.join(base, 'brdf'), exist_ok=True)
        os.makedirs(osp.join(base, 'error'), exist_ok=True)

        #* plot the ltc evaluation results
        fig = plt.figure(figsize=(10.24, 10.24))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(-vec[0][0], vec[0][1], vec[0][2], rstride = 1, cstride = 1, facecolors=cm.coolwarm(evalLTC.numpy()))
        plt.savefig(osp.join(base, 'ltc', file+'.png'))
        plt.close()
        
        #* plot the brdf evaluation results
        fig = plt.figure(figsize=(10.24, 10.24))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(-vec[0][0], vec[0][1], vec[0][2], rstride = 1, cstride = 1, facecolors=cm.coolwarm(evalBRDF.numpy()))
        plt.savefig(osp.join(base, 'brdf', file+'.png'))
        plt.close()

        #* plot the error between evaluation of ltc and brdf
        fig = plt.figure(figsize=(10.24, 10.24))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(-vec[0][0], vec[0][1], vec[0][2], rstride = 1, cstride = 1, facecolors=cm.coolwarm(torch.abs(evalLTC-evalBRDF).numpy()))
        plt.savefig(osp.join(base, 'error', file+'.png'))
        plt.close()
    else:
        #* define the name of output images
        base, file = osp.split(name)
        os.makedirs(base, exist_ok=True)
        #* plot the Do evaluation results
        fig = plt.figure(figsize=(40.96, 10.24))
        ax = fig.add_subplot(141, projection='3d')
        if rect is not None:
            ax = plotVec(ax, (ltc.invM @ rect.T).T)
            ax.set_xlim(-1.5,1.5)
            ax.set_ylim(-1.5,1.5)
            ax.set_zlim(-1.5,1.5)
        ax.plot_surface(-vec[0][0], vec[0][1], vec[0][2], rstride = 1, cstride = 1, facecolors=cm.coolwarm(evalDo.numpy()))

        #* plot the ltc evaluation results
        ax = fig.add_subplot(142, projection='3d')
        if rect is not None:
            ax = plotVec(ax, rect)
            ax.set_xlim(-1.5,1.5)
            ax.set_ylim(-1.5,1.5)
            ax.set_zlim(-1.5,1.5)
        ax.plot_surface(-vec[0][0], vec[0][1], vec[0][2], rstride = 1, cstride = 1, facecolors=cm.coolwarm(evalLTC.numpy()))
        
        #* plot the brdf evaluation results
        ax = fig.add_subplot(143, projection='3d')
        ax.plot_surface(-vec[0][0], vec[0][1], vec[0][2], rstride = 1, cstride = 1, facecolors=cm.coolwarm(evalBRDF.numpy()))
        if rect is not None:
            ax.set_xlim(-1.5,1.5)
            ax.set_ylim(-1.5,1.5)
            ax.set_zlim(-1.5,1.5)

        #* plot the error between evaluation of ltc and brdf
        ax = fig.add_subplot(144, projection='3d')
        ax.plot_surface(-vec[0][0], vec[0][1], vec[0][2], rstride = 1, cstride = 1, facecolors=cm.coolwarm(torch.abs(evalLTC-evalBRDF).numpy()), alpha=0.2)
        plt.savefig(osp.join(base, file+'.png'))
        plt.close()

def fit(theta=60, roughness=1.0, Z=None, initialGuess=None, log=False):
    thetaArc = theta/180*np.pi
    viewDir = np.array([np.sin(thetaArc), 0, np.cos(thetaArc)], dtype=np.float32).reshape(1, 3, 1, 1)

    # * rotation matrix and scale matrix initialization
    brdfArgs = {}
    brdfArgs['size'] = 256
    brdf = PlanarSVBRDF(brdfArgs)
    if Z is None:
        Z = brdf.avgVec(thetaView=theta, roughness=roughness)
    Y = np.array([0, 1, 0], dtype=np.float32)
    X = np.array([Z[2], 0.0, -Z[0]], dtype=np.float32)
    brdfNorm, fresNorm = brdf.norm(thetaView=theta, roughness=roughness, fresNorm=True)
    if initialGuess is None:
        initialGuess = [1.0, 1.0, 0.0, 0.0]
    # initialGuess = [ 0.82388573,  0.55209155, -0.09494477, -0.33613882]
    n = np.array([0, 0, 1], dtype=np.float32).reshape(1, 3, 1, 1)

    ltc = LTC(X=X, Y=Y, Z=Z, m00=initialGuess[0], m11=initialGuess[1], m02=initialGuess[2], m20=initialGuess[3], amplitude=brdfNorm)

    res = optimize.minimize(computeError, initialGuess, (ltc, brdf, roughness, viewDir, n, log), method='Nelder-Mead')
    return res, ltc, brdf, fresNorm


if __name__ == "__main__":
    theta = 0.23116
    roughness = -0.6
    if False:
        res, ltc, brdf, fresNorm = fit(theta=theta, roughness=roughness, log=True)
        print(res)
    else:
        lut = torch.load('deepmaterial/utils/LTC/look-up-table.pth')
        nSample = 64
        t = round(theta / np.pi * 2 * (nSample-1))
        a = round((roughness/2+0.5) * (nSample-1))
        params = lut[a, t]
        ltc = LTC(m00=params[...,0], m11=params[...,1], m02=params[...,2], m20=params[...,3], amplitude=params[...,4], mode='eval')
        brdfArgs = {}
        brdfArgs['size'] = 256
        brdfArgs['nbRendering'] = 1
        brdf = PlanarSVBRDF(brdfArgs)
        # rect = RectLighting(brdfArgs, [0, 0, 1.5], [1, 0, 0], [0, 1, 0], 10/20, 8/20)

    plotResult(ltc, brdf, roughness=roughness, theta=theta/np.pi * 180, jointPlot=True, name='tmp/result')