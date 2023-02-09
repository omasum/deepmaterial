import numpy as np
import scipy.optimize as optimize
from matplotlib import cm, colors
import matplotlib.pyplot as plt

# np.maximum(Wi[2], 0)
fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
def spherical_dir(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def meshgrid_spherical_coord(numsamples, hemisphere=False):
    theta = np.linspace(0, np.pi, num=numsamples)
    if hemisphere:
        theta = np.linspace(0, np.pi/2, num=numsamples)
    phi = np.linspace(0, 2*np.pi, num=numsamples*2)
    theta, phi = np.meshgrid(theta, phi)
    return theta, phi

def spherical_integral(integrand, num_samples=128, hemisphere=False):
    theta_max = np.pi
    if hemisphere:
        theta_max = np.pi * 0.5

    phi_max = np.pi * 2

    theta = np.linspace(0, theta_max, num_samples)
    phi = np.linspace(0, phi_max, num_samples)
    theta, phi = np.meshgrid(theta, phi)
    vec = np.array(spherical_dir(theta, phi))

    v = integrand(vec)
    dim = len(v.shape)
    integral = np.sum(v * np.sqrt(1 - vec[2] ** 2), axis=(dim-2, dim-1)) * theta_max * phi_max / num_samples / num_samples
    return integral


def D_o(Wi):
    return np.maximum(Wi[2], 0) / np.pi

def D_ltc(Wi, transfo):
    #Wi is the expected polygonal lighting range
    # transfo is the transformation matrix M
    x = Wi[0]
    y = Wi[1]
    z = Wi[2]
    
    inv_transfo = np.linalg.inv(transfo)
    
    # w*M^-1
    x_orig = x * inv_transfo[0][0] + y * inv_transfo[0][1] + z * inv_transfo[0][2]
    y_orig = x * inv_transfo[1][0] + y * inv_transfo[1][1] + z * inv_transfo[1][2]
    z_orig = x * inv_transfo[2][0] + y * inv_transfo[2][1] + z * inv_transfo[2][2]

    l = np.sqrt(x_orig**2 + y_orig**2 + z_orig**2)

    # vector normalization
    x_orig /= l
    y_orig /= l
    z_orig /= l
    # w_o = [xorig, yorig, zorig]

    # evaluate consine spherical function
    wi_orig = np.array([x_orig, y_orig, z_orig])
    vals = D_o(wi_orig)
    
    # apply change of variable
    jacobian = np.linalg.det(inv_transfo) / (l*l*l);
    vals *= jacobian
    
    return vals

def Gvis_GGX(NdotV, NdotL, alpha):
    a2 = alpha*alpha
    G_V = NdotV + np.sqrt( (NdotV - NdotV * a2) * NdotV + a2 )
    G_L = NdotL + np.sqrt( (NdotL - NdotL * a2) * NdotL + a2 )
    return 1.0/( G_V * G_L )

def ggx_cbrdf(theta, roughness, wi):
    wo = np.array([-np.sin(theta), 0, np.cos(theta)])
    wh = np.array([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]])
    wh /= np.sqrt(wh[0]**2 + wh[1]**2 + wh[2]**2)

    NoH = np.maximum(wh[2], 0)
    NoV = np.maximum(wo[2], 0)
    NoL = np.maximum(wi[2], 0)

    alpha = roughness * roughness
    a2 = alpha * alpha
    d = ((NoH * a2 - NoH) * NoH + 1)
    NDF = a2 / (np.pi * d * d)
    G = Gvis_GGX(NoV, NoL, alpha)
    return NDF * G *  NoL

theta_view = 60/180*np.pi
roughness = np.sqrt(0.36)
ggx_norm = spherical_integral(lambda wi: ggx_cbrdf(theta_view, roughness, wi), num_samples=1024, hemisphere=True)

def ggx_transfo_matrix(m00, m11, m02, z_axis):
    scale_mat = np.identity(3)
    scale_mat = np.array([[m00, 0, m02],
                          [0, m11, 0],
                          [0, 0, 1]])

    rotate_mat = np.identity(3)
    reflect_vec = np.array([np.sin(theta_view), 0, np.cos(theta_view)])
    # reflect_vec = z_axis
    Z = reflect_vec
    Y = np.array([0, 1, 0])
    X = np.cross(Y, Z)
    X /= np.linalg.norm(X)
#     rotate_mat[0] = X
#     rotate_mat[1] = Y
#     rotate_mat[2] = Z
    rotate_mat = np.array([X, Y, Z])
    rotate_mat = np.transpose(rotate_mat)

    transfo = rotate_mat @ scale_mat
    return transfo

def ggx_approx(m00, m11, m02, norm, z_axis):
    def f(wi):
        transfo = ggx_transfo_matrix(m00, m11, m02, z_axis)
        return D_ltc(wi, transfo) * norm
    return f
    
def ggx_cbrdf_dominant_dir(theta, roughness, wi):
    wo = np.array([-np.sin(theta), 0, np.cos(theta)])
    wh = np.array([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]])
    wh /= np.sqrt(wh[0]**2 + wh[1]**2 + wh[2]**2)

    NoH = np.maximum(wh[2], 0)
    NoV = np.maximum(wo[2], 0)
    NoL = np.maximum(wi[2], 0)

    alpha = roughness * roughness
    a2 = alpha * alpha
    d = ((NoH * a2 - NoH) * NoH + 1)
    NDF = a2 / (np.pi * d * d)
    G = Gvis_GGX(NoV, NoL, alpha)
    ret = NDF * G *  NoL
    ind = np.unravel_index(np.argmax(ret, axis=None), ret.shape)
    print("max index: ", ind)
    direction = np.array([wi[0][ind], wi[1][ind], wi[2][ind]])
    print("max value: ", ret[ind], np.max(ret))
    return direction

theta, phi = meshgrid_spherical_coord(256, hemisphere=True)
x, y, z = spherical_dir(theta, phi)
wi = np.array([x, y, z])

ggx_dir_x = spherical_integral(lambda wi: ggx_cbrdf(theta_view, roughness, wi) * wi[0],
                             num_samples=1024,
                             hemisphere=True)
ggx_dir_z = spherical_integral(lambda wi: ggx_cbrdf(theta_view, roughness, wi) * wi[2],
                             num_samples=1024,
                             hemisphere=True)
ggx_dir = np.array([ggx_dir_x, 0, ggx_dir_z]) # The average direction of ggx which doesn' always intersect the texture plane.

ggx_dir = ggx_cbrdf_dominant_dir(theta_view, roughness, wi) # Orthonormal projection of the shading point onto the texture plane.

def obj_fn(p):
    m00, m11, m02 = p
    f = ggx_approx(m00, m11, m02, ggx_norm, ggx_dir)
    # norm = spherical_integral(lambda wi: f(wi), num_samples=1024, hemisphere=True)
    # f = ggx_approx(m00, m11, m20, ggx_norm/norm)

    theta, phi = meshgrid_spherical_coord(1024, hemisphere=True)
    x, y, z = spherical_dir(theta, phi)
    vals_approx = f(np.array([x, y, z])) # The distribution translated from cosine sphere distribution
    vals_exact = ggx_cbrdf(theta_view, roughness, np.array([x, y, z]))

    diff = np.abs(vals_exact - vals_approx)**2
    return np.mean(diff)

initial_guess = [0.6301864 ,  0.4180192 , 0]
result = optimize.minimize(obj_fn, initial_guess, method="Nelder-Mead")


def plot_ltc_func(transfo):
    theta, phi = meshgrid_spherical_coord(64)
    x, y, z = spherical_dir(theta, phi)
    
    vals = D_ltc(np.array([x, y, z]), transfo)

    # normalize the value for better visualization
    vals /= np.max(vals)
    ax.plot_surface(x, z, y, wireframe=False, color=cm.coolwarm(vals))
#     ipv.plot_mesh(x*vals, z*vals, y*vals, wireframe=False, color=cm.coolwarm(vals))

    # plot lines in order to visualize the geometry transform
    sample_points = np.array([[0, 0, 2], [-1, -1, 2], [-1, 1, 2], [1, 1, 2], [1, -1, 2]])
    for p in sample_points:
        p = np.dot(transfo, p)
        p /= np.linalg.norm(p)
        p *= 2
        fig.plot([0, p[0]], [0, p[2]], [0, p[1]])


def plot(a, b, c, d):
    transfo = np.identity(3, dtype = float)        
    transfo[0][0] = a
    transfo[0][2] = b
    transfo[1][1] = c
    transfo[2][0] = d

    fig.meshes.clear()
    fig.scatters.clear()
    plot_ltc_func(transfo)
    fig.xyzlim(-2, 2)

print(result)