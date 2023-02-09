import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from deepmaterial.utils.LTC.ltc_lut import lut_size, ltc_matrix, ltc_amplitude

def sampleLut(roughness, costheta):
    uvscale = (lut_size - 1.0) / lut_size
    uvbias = 0.5 / lut_size
    # calculate the uv index of ltc_matrix.
    uv = np.array([roughness, costheta]) * uvscale + uvbias
    st = uv * lut_size
    iuv = np.floor(st)
    fuv = st - iuv
    
    # indexing four corner matrix, and interpolate by lerp
    a = ltc_matrix[int(iuv[1]), int(iuv[0])]
    b = ltc_matrix[int(iuv[1]), np.minimum(63, int(iuv[0] + 1))]
    c = ltc_matrix[np.minimum(63, int(iuv[1] + 1)), int(iuv[0])]
    d = ltc_matrix[np.minimum(63, int(iuv[1] + 1)), np.minimum(63, int(iuv[0]) + 1)]
    lerp = lambda t, a, b: (1.0 - t) * a + t * b
    M = lerp(fuv[1], lerp(fuv[0], a, b), lerp(fuv[0], c, d))
    M = np.transpose(M)
    return M, np.linalg.inv(M)

def evalLtc(L, M, invM):
    Loriginal = np.dot(invM, L)
    Loriginal = Loriginal / np.linalg.norm(Loriginal)

    L_ = np.dot(M, Loriginal)

    l = np.linalg.norm(L_)
    Jacobian = np.linalg.det(M) / (l*l*l);

    D = np.maximum(0.0, Loriginal[2])
    return D / Jacobian

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
    
def plot(roughness, theta):
    grid_res_x = 256
    grid_res_y = grid_res_x

    # convention of scipy: theta is the azimuthal angle and phi is the polar angle
    phiSeq = np.linspace(0, np.pi, num=grid_res_x)
    thetaSeq = np.linspace(0, 2*np.pi, num=grid_res_y)
    phiSeq, thetaSeq = np.meshgrid(phiSeq, thetaSeq)

    x = np.sin(phiSeq) * np.cos(thetaSeq)
    y = np.sin(phiSeq) * np.sin(thetaSeq)
    z = np.cos(phiSeq)

    # clamped cosine lobe
    dist = np.maximum(z, 0)

    M, invM = sampleLut(roughness, np.cos(theta))
    for row in range(grid_res_y):
        for colum in range(grid_res_x):
            L = [x[row][colum], y[row][colum], z[row][colum]]
            dist[row][colum] = evalLtc(L, M, invM)

    normalized_dist = dist / np.maximum(np.nanmax(dist), 1.0)
    ax.plot_surface(-x, y, z, facecolors=cm.jet(normalized_dist), rcount=grid_res_x, ccount=grid_res_y)
plot(0.4, np.pi/6)
plt.savefig('tmp/img.png')
# a = widgets.FloatSlider(min=0,max=1,step=0.1,value=0.8)
# t = widgets.FloatSlider(min=0,max=np.pi/2,step=np.pi/200,value=np.pi/4)
# widgets.interact(plot, roughness=a, theta=t)
