import matplotlib
from matplotlib import cm, colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# import jax.numpy as np
import scipy.optimize as optimize

coolwarm_cmap = matplotlib.cm.get_cmap('coolwarm')
coolwarm_rgb = []
norm = matplotlib.colors.Normalize(vmin=0, vmax=255)

for i in range(0, 255):
       k = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(i)))
       coolwarm_rgb.append(k)

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = np.array(cmap(k*h)[:3])*255
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

coolwarm_plotly = matplotlib_to_plotly(coolwarm_cmap, 255)

def spherical_dir(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def spherical_coord(x, y, z):
    norm = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/norm)
    phi = np.arctan2(y, x)
    return theta, phi

def meshgrid_spherical_coord(numsamples, hemisphere=False):
    theta = np.linspace(0, np.pi, num=numsamples)
    if hemisphere:
        theta = np.linspace(0, np.pi/2, num=numsamples)
    phi = np.linspace(0, 2*np.pi, num=numsamples*2)
    theta, phi = np.meshgrid(theta, phi)
    return theta, phi

def D(Wi):
    return np.maximum(Wi[2], 0) / np.pi

def D_ltc(Wi, transfo):
    x = Wi[0]
    y = Wi[1]
    z = Wi[2]
    
    inv_transfo = np.linalg.inv(transfo)
    
    x_orig = x * inv_transfo[0][0] + y * inv_transfo[0][1] + z * inv_transfo[0][2]
    y_orig = x * inv_transfo[1][0] + y * inv_transfo[1][1] + z * inv_transfo[1][2]
    z_orig = x * inv_transfo[2][0] + y * inv_transfo[2][1] + z * inv_transfo[2][2]

    l = np.sqrt(x_orig**2 + y_orig**2 + z_orig**2)

    # vector normalization
    x_orig /= l
    y_orig /= l
    z_orig /= l

    # evaluate spherical function
    wi_orig = np.array([x_orig, y_orig, z_orig])
    vals = D(wi_orig)
    
    # apply change of variable
    jacobian = np.linalg.det(inv_transfo) / (l*l*l);
    vals *= jacobian
    
    return vals

def plot_ltc_func(transfo, add_trace, scale_by_value=False):
    numsamples = 64
    theta_ = np.linspace(0, np.pi, num=numsamples)
    phi_ = np.linspace(0, 2*np.pi, num=numsamples*2)
    theta, phi = np.meshgrid(theta_, phi_)

    x, y, z = spherical_dir(theta, phi)
    
    vals = D_ltc(np.array([x, y, z]), transfo)

    # normalize the value for better visualization
    vals /= np.max(vals)

    if (scale_by_value):
        add_trace(go.Surface(x=x*vals, y=y*vals, z=z*vals,
                             surfacecolor=vals,
                             colorscale=coolwarm_plotly,
                             showscale=False))
    else:
        add_trace(go.Surface(x=x, y=y, z=z,
                             surfacecolor=vals,
                             colorscale=coolwarm_plotly,
                             showscale=False))

    # ipv.plot_mesh(x, z, y, wireframe=False, color=cm.coolwarm(vals))
    # ipv.plot_mesh(x*vals, z*vals, y*vals, wireframe=False, color=cm.coolwarm(vals))

    # plot lines in order to visualize the geometry transform
    sample_points = np.array([[0, 0, 2], [-1, -1, 2], [-1, 1, 2], [1, 1, 2], [1, -1, 2]])
    for p in sample_points:
        p = np.dot(transfo, p)
        p /= np.linalg.norm(p)
        p *= 1.5
        add_trace(go.Scatter3d(x=[0, p[0]], y=[0, p[1]], z=[0, p[2]],
                               mode='lines',
                               showlegend=False,
                               line=dict(color='black')))

    # axis_dic = dict(backgroundcolor="rgba(0, 0, 0, 0)",
    #                 visible=False)
    # fig.update_layout(scene=dict(xaxis=axis_dic,
    #                              yaxis=axis_dic,
    #                              zaxis=axis_dic))
    # return fig


def plot(a, b, c, d, e, add_trace, scale_by_value=False):
#     transfo = np.identity(3, dtype = float)
#     transfo[0][0] = a
#     transfo[0][2] = b
#     transfo[1][1] = c
#     transfo[2][0] = d
#     transfo[2][2] = e
    transfo = np.array([[a, 0, b],
                        [0, c, 0],
                        [d, 0, e]])

    plot_ltc_func(transfo, add_trace, scale_by_value=scale_by_value)
    # setup_ipv_viev()
    # ipv.show()

fig = make_subplots(rows=2, cols=6,
                    specs=[[{'is_3d': True} for _ in range(6)],[{'is_3d': True} for _ in range(6)]],
                    subplot_titles=['m00=0.8', 'm00=0.4', 'm11=0.8', 'm11=0.4','m22=1.2', 'm22=2.0'],
                    horizontal_spacing = 0.01, vertical_spacing = 0.05)

plot(0.8, 0, 1, 0, 1, lambda c: fig.add_trace(c, row=1, col=1))
plot(0.4, 0, 1, 0, 1, lambda c: fig.add_trace(c, row=1, col=2))
plot(1, 0, 0.8, 0, 1, lambda c: fig.add_trace(c, row=1, col=3))
plot(1, 0, 0.4, 0, 1, lambda c: fig.add_trace(c, row=1, col=4))
plot(1, 0, 1, 0, 1.2, lambda c: fig.add_trace(c, row=1, col=5))
plot(1, 0, 1, 0, 2.0, lambda c: fig.add_trace(c, row=1, col=6))
plot(0.8, 0, 1, 0, 1, lambda c: fig.add_trace(c, row=2, col=1), scale_by_value=True)
plot(0.4, 0, 1, 0, 1, lambda c: fig.add_trace(c, row=2, col=2), scale_by_value=True)
plot(1, 0, 0.8, 0, 1, lambda c: fig.add_trace(c, row=2, col=3), scale_by_value=True)
plot(1, 0, 0.4, 0, 1, lambda c: fig.add_trace(c, row=2, col=4), scale_by_value=True)
plot(1, 0, 1, 0, 1.2, lambda c: fig.add_trace(c, row=2, col=5), scale_by_value=True)
plot(1, 0, 1, 0, 2.0, lambda c: fig.add_trace(c, row=2, col=6), scale_by_value=True)

axis_dict = dict(backgroundcolor="rgba(0, 0, 0, 0)", visible=False)
camera_dict = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0.8, y=1.2, z=0.8)
)
fig.update_layout(width=3072, height=1024)
for k, i in enumerate(fig['layout']['annotations']):
    if k < 6:
        i['font'] = dict(size=40)
    else:
        i['font'] = dict(size=0)
for k in range(12):
    scene = "scene"
    if k > 0:
        scene += str(k+1)
    camera = scene + "_camera"
    fig['layout'][scene]['xaxis'] = axis_dict
    fig['layout'][scene]['yaxis'] = axis_dict
    fig['layout'][scene]['zaxis'] = axis_dict
    fig['layout'][camera] = camera_dict

# for k, i in enumerate(fig['layout']['scene']):
#     print(i)
#     i['xaxis']['backgroundcolor'] = "rgba(0, 0, 0, 0)"
fig.show()
fig = make_subplots(rows=2, cols=6,
                    specs=[[{'is_3d': True} for _ in range(6)], [{'is_3d': True} for _ in range(6)]],
                    subplot_titles=['m02=0.1', 'm02=0.5', 'm02=0.9', 'm20=0.1','m20=0.5', 'm20=0.9'],
                   horizontal_spacing = 0.01, vertical_spacing = 0.05)

plot(1, 0.1, 1, 0, 1, lambda c: fig.add_trace(c, row=1, col=1))
plot(1, 0.5, 1, 0, 1, lambda c: fig.add_trace(c, row=1, col=2))
plot(1, 0.9, 1, 0, 1, lambda c: fig.add_trace(c, row=1, col=3))
plot(1, 0, 1, 0.1, 1, lambda c: fig.add_trace(c, row=1, col=4))
plot(1, 0, 1, 0.5, 1, lambda c: fig.add_trace(c, row=1, col=5))
plot(1, 0, 1, 0.9, 1, lambda c: fig.add_trace(c, row=1, col=6))
plot(1, 0.1, 1, 0, 1, lambda c: fig.add_trace(c, row=2, col=1), scale_by_value=True)
plot(1, 0.5, 1, 0, 1, lambda c: fig.add_trace(c, row=2, col=2), scale_by_value=True)
plot(1, 0.9, 1, 0, 1, lambda c: fig.add_trace(c, row=2, col=3), scale_by_value=True)
plot(1, 0, 1, 0.1, 1, lambda c: fig.add_trace(c, row=2, col=4), scale_by_value=True)
plot(1, 0, 1, 0.5, 1, lambda c: fig.add_trace(c, row=2, col=5), scale_by_value=True)
plot(1, 0, 1, 0.9, 1, lambda c: fig.add_trace(c, row=2, col=6), scale_by_value=True)

axis_dict = dict(backgroundcolor="rgba(0, 0, 0, 0)", visible=False)
camera_dict = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0.8, y=1.2, z=0.8)
)
camera2_dict = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=2, z=0)
)

fig.update_layout(width=3072, height=1024)
for k, i in enumerate(fig['layout']['annotations']):
    if k < 6:
        i['font'] = dict(size=40)
    else:
        i['font'] = dict(size=0)
for k in range(12):
    scene = "scene"
    if k > 0:
        scene += str(k+1)
    camera = scene + "_camera"
    fig['layout'][scene]['xaxis'] = axis_dict
    fig['layout'][scene]['yaxis'] = axis_dict
    fig['layout'][scene]['zaxis'] = axis_dict
    if k < 6:
        fig['layout'][camera] = camera_dict
    else:
        fig['layout'][camera] = camera2_dict

fig.show()

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

# Has cosine builtin
def phong_cbrdf(theta, roughness, wi):
    wo = np.array([-np.sin(theta), 0, np.cos(theta)])
    r = wo
    r = np.array([-r[0], r[1], r[2]])
    
    cosThetaH = np.maximum((wi[0]*r[0] + wi[1]*r[1] + wi[2]*r[2]), 0)
    alpha = roughness * roughness
    a2 = alpha * alpha
    power = 2/a2 - 2
    norm = 1.0/np.pi/a2
    return norm * np.power(cosThetaH, power) * np.maximum(wi[2], 0)

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

def getSpecularDominantDir(N, R, alpha):
    a2 = alpha * alpha;
    return R * (1-a2) + N*a2;

    smoothness = 1 - alpha
    factor = smoothness * (np.sqrt(smoothness) + alpha)
    return N * (1 - factor) + R * factor


theta_view = 60/180*np.pi
roughness = np.sqrt(0.36)

theta, phi = meshgrid_spherical_coord(256, hemisphere=True)
x, y, z = spherical_dir(theta, phi)
wi = np.array([x, y, z])

phong_norm = spherical_integral(lambda wi: phong_cbrdf(theta_view, roughness, wi), num_samples=1024, hemisphere=True)
print("Phong normalization: ", phong_norm)
vals = phong_cbrdf(theta_view, roughness, wi)

ggx_norm = spherical_integral(lambda wi: ggx_cbrdf(theta_view, roughness, wi), num_samples=1024, hemisphere=True)

def ggx_cbrdf_average_term(wi):
    v = ggx_cbrdf(theta_view, roughness, wi)
    return v

ggx_dir_x = spherical_integral(lambda wi: ggx_cbrdf(theta_view, roughness, wi) * wi[0],
                             num_samples=1024,
                             hemisphere=True)
ggx_dir_z = spherical_integral(lambda wi: ggx_cbrdf(theta_view, roughness, wi) * wi[2],
                             num_samples=1024,
                             hemisphere=True)
ggx_dir = np.array([ggx_dir_x, 0, ggx_dir_z])

ggx_dir = ggx_cbrdf_dominant_dir(theta_view, roughness, wi)

# ggx_dir_y = spherical_integral(lambda wi: ggx_cbrdf_average_term(wi) * wi[1],
#                              num_samples=1024,
#                              hemisphere=True)
# ggx_dir_z = spherical_integral(lambda wi: ggx_cbrdf_average_term(wi) * wi[2],
#                              num_samples=1024,
#                              hemisphere=True)
# ggx_dir = np.array([ggx_dir_x, ggx_dir_y, ggx_dir_z])
reflect_dir = np.array([np.sin(theta_view), 0, np.cos(theta_view)])
# ggx_dir = getSpecularDominantDir(np.array([0,0,1]), reflect_dir, roughness*roughness)
# ggx_dir /= np.linalg.norm(ggx_dir)
print("ggx dir ", ggx_dir)
# ggx_dir = np.array([0.625941455, 0, 0.779869974])

print("GGX normalization: ", ggx_norm)
ggx_vals = ggx_cbrdf(theta_view, roughness, wi)
phong_vals = phong_cbrdf(theta_view, roughness, wi)

def ggx_approx(m00, m11, m02, norm, z_axis):
    def f(wi):
        transfo = ggx_transfo_matrix(m00, m11, m02, z_axis)
        return D_ltc(wi, transfo) * norm
    return f

ggx_cbrdf_approx = ggx_approx(0.79849551,  0.38944701, -0.14630333, ggx_norm, ggx_dir)
ggx_norm_approx = spherical_integral(lambda wi: ggx_cbrdf_approx(wi), num_samples=1024, hemisphere=False)
print("GGX LTC normalization: ", ggx_norm_approx)
# ggx_cbrdf_approx = ggx_approx(1.02277543,  0.621345  , -0.1516238, ggx_norm/ggx_norm_approx)
# print("GGX LTC normalization: ", spherical_integral(lambda wi: ggx_cbrdf_approx(wi), num_samples=1024, hemisphere=False))
ggx_vals_approx = ggx_cbrdf_approx(wi)

# normalize the value for better visualization
# vals /= np.max(vals)

max_val = np.max(ggx_vals)

# max_val = np.maximum(max_val, np.max(ggx_vals_approx))


fig = go.Figure(data=go.Surface(x=x*ggx_vals, y=y*ggx_vals, z=z*ggx_vals,
                                colorscale=coolwarm_plotly,
                                surfacecolor=ggx_vals,
                                opacity=0.6))

fig.add_trace(go.Surface(x=x*ggx_vals_approx, y=y*ggx_vals_approx, z=z*ggx_vals_approx,
                                colorscale=coolwarm_plotly,
                                surfacecolor=ggx_vals_approx,
                                opacity=0.6))

fig.add_trace(go.Scatter3d(x=[0, np.sin(theta_view) * 6], y=[0, 0], z=[0, np.cos(theta_view) * 6],
                           mode='lines',
                           showlegend=False,
                           line=dict(color='black')))
fig.add_trace(go.Scatter3d(x=[0, ggx_dir[0]*6], y=[0, ggx_dir[1]*6], z=[0, ggx_dir[2]*6],
                           mode='lines',
                           showlegend=False,
                           line=dict(color='red')))

fig.add_trace(go.Scatter3d())
fig.update_layout(scene=dict(xaxis=dict(range=[-max_val,max_val]),
                             yaxis=dict(range=[-max_val,max_val]),
                             zaxis=dict(range=[-max_val,max_val]),
                             aspectmode='manual',
                             aspectratio=dict(x=1, y=1, z=1)))
fig.show()

def phong_transfo_matrix(a):
    scale_mat = np.identity(3)
    scale_mat[0][0] = a
    scale_mat[1][1] = a
#     scale_mat = np.array([[a, 0, 0],
#                           [0, a, 0],
#                           [0, 0, 1]])

    rotate_mat = np.identity(3)
    reflect_vec = np.array([np.sin(theta_view), 0, np.cos(theta_view)])
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

def phong_approx(a, norm):
    def f(wi):
        transfo = phong_transfo_matrix(a)
        return D_ltc(wi, transfo) * norm
    return f

def ggx_transfo_matrix(m00, m11, m02, z_axis):
    scale_mat = np.identity(3)
    scale_mat = np.array([[m00, 0, m02],
                          [0, m11, 0],
                          [0, 0, 1]])

    rotate_mat = np.identity(3)
    reflect_vec = np.array([np.sin(theta_view), 0, np.cos(theta_view)])
    reflect_vec = z_axis
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

ggx_transfo_matrix(0.63028784,  0.41804157, -0.07809715, ggx_dir)

fig = make_subplots(rows=1, cols=5,
                    specs=[[{'is_3d': True} for _ in range(5)]],
                    subplot_titles=['Cosine函数', '对m00,m11进行缩放', '对m02进行偏移', '以BRDF主要方向为Z轴', 'GGX BRDF'],
                    horizontal_spacing = 0.01, vertical_spacing = 0.05)

fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=z, colorscale=coolwarm_plotly, showscale=False), row=1, col=1)

ggx_cbrdf_approx = ggx_approx(0.63028784,  0.41804157, 0, ggx_norm, np.array([0.0,0.0,1.0]))
ggx_vals_approx = ggx_cbrdf_approx(wi)
fig.add_trace(go.Surface(x=x, y=y, z=z,
                         surfacecolor=ggx_vals_approx,
                         colorscale=coolwarm_plotly,
                         showscale=False,
                         opacity=1.0),
              row=1, col=2)

ggx_cbrdf_approx = ggx_approx(0.63028784,  0.41804157, -0.07809715, ggx_norm, np.array([0.0,0.0,1.0]))
ggx_vals_approx = ggx_cbrdf_approx(wi)
fig.add_trace(go.Surface(x=x, y=y, z=z,
                         surfacecolor=ggx_vals_approx,
                         colorscale=coolwarm_plotly,
                         showscale=False,
                         opacity=1.0),
              row=1, col=3)

ggx_cbrdf_approx = ggx_approx(0.63028784,  0.41804157, -0.07809715, ggx_norm, ggx_dir)
ggx_vals_approx = ggx_cbrdf_approx(wi)
fig.add_trace(go.Surface(x=x, y=y, z=z,
                         surfacecolor=ggx_vals_approx,
                         colorscale=coolwarm_plotly,
                         showscale=False,
                         opacity=1.0),
              row=1, col=4)

ggx_vals = ggx_cbrdf(theta_view, roughness, wi)
fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=ggx_vals, showscale=False, colorscale=coolwarm_plotly),
              row=1, col=5)

fig.update_layout(width=2560, height=512)
for k, i in enumerate(fig['layout']['annotations']):
    i['font'] = dict(size=40)

for k in range(5):
    scene = "scene"
    if k > 0:
        scene += str(k+1)
    fig['layout'][scene]['xaxis'] = axis_dict
    fig['layout'][scene]['yaxis'] = axis_dict
    fig['layout'][scene]['zaxis'] = axis_dict

# max_val = np.max(vals)
# fig.update_layout(scene=dict(xaxis=dict(range=[-max_val,max_val]),
#                              yaxis=dict(range=[-max_val,max_val]),
#                              zaxis=dict(range=[-max_val,max_val]),
#                              aspectmode='manual',
#                              aspectratio=dict(x=1, y=1, z=1)))
fig.show()

fig = make_subplots(rows=1, cols=4,
                    specs=[[{'is_3d': True} for _ in range(4)]],
                    subplot_titles=['Cosine函数', '对m00,m11进行缩放', '以反射向量为Z轴', 'GGX BRDF'],
                    horizontal_spacing = 0.01, vertical_spacing = 0.05)

fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=z, colorscale=coolwarm_plotly, showscale=False), row=1, col=1)

theta_view = 0/180*np.pi
ltc_rotated = phong_approx(0.38, phong_norm)
vals = ltc_rotated(wi)

# ltc_norm = spherical_integral(lambda wi: ltc_rotated(wi), num_samples=4096, hemisphere=True)
# print("ltc normalization: ", ltc_norm)

# normalize the value for better visualization
# vals /= np.max(vals)
fig.add_trace(go.Surface(x=x, y=y, z=z,
                         surfacecolor=vals,
                         colorscale=coolwarm_plotly,
                         showscale=False,
                         opacity=1.0),
              row=1, col=2)

theta_view = 30/180*np.pi
vals_approx = phong_approx(0.38, phong_norm)(wi)
fig.add_trace(go.Surface(x=x, y=y, z=z,
                         surfacecolor=vals_approx,
                         colorscale=coolwarm_plotly,
                         showscale=False,
                         opacity=1.0),
              row=1, col=3)

r_vals = phong_cbrdf(theta_view, roughness, wi)
fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=r_vals, showscale=False, colorscale=coolwarm_plotly),
              row=1, col=4)

fig.update_layout(width=2048, height=512)
for k, i in enumerate(fig['layout']['annotations']):
    i['font'] = dict(size=40)

for k in range(4):
    scene = "scene"
    if k > 0:
        scene += str(k+1)
    fig['layout'][scene]['xaxis'] = axis_dict
    fig['layout'][scene]['yaxis'] = axis_dict
    fig['layout'][scene]['zaxis'] = axis_dict

# max_val = np.max(vals)
# fig.update_layout(scene=dict(xaxis=dict(range=[-max_val,max_val]),
#                              yaxis=dict(range=[-max_val,max_val]),
#                              zaxis=dict(range=[-max_val,max_val]),
#                              aspectmode='manual',
#                              aspectratio=dict(x=1, y=1, z=1)))
fig.show()

def obj_fn(p):
    m00, m11, m02 = p
    f = ggx_approx(m00, m11, m02, ggx_norm, ggx_dir)
    # norm = spherical_integral(lambda wi: f(wi), num_samples=1024, hemisphere=True)
    # f = ggx_approx(m00, m11, m20, ggx_norm/norm)

    theta, phi = meshgrid_spherical_coord(512, hemisphere=True)
    x, y, z = spherical_dir(theta, phi)
    vals_approx = f(np.array([x, y, z]))
    vals_exact = ggx_cbrdf(theta_view, roughness, np.array([x, y, z]))

    diff = np.abs(vals_exact - vals_approx)**2
    return np.mean(diff)

initial_guess = [0.6301864 ,  0.4180192 , 0]
result = optimize.minimize(obj_fn, initial_guess, method="Nelder-Mead")

print(result)
# result.x = 0.443
# print(ggx_transfo_matrix(result.x))
# f = phong_approx(result.x)
# theta, phi = meshgrid_spherical_coord(64)
# x, y, z = spherical_dir(theta, phi)
# vals = f(np.array([x, y, z]))
# vals_exact = phong_cbrdf(theta_view, roughness, np.array([x, y, z]))
# # normalize the value for better visualization
# vals = vals - vals_exact
# # vals /= np.max(vals)
# ipv.clear()
# ipv.plot_mesh(x*vals, z*vals, y*vals, wireframe=False, color=cm.coolwarm(vals))
# setup_ipv_viev()
# ipv.show()

import jax


def loss(params):
    a = params
    return obj_fn(a)

def update_parameters_step(params, learning_rate=0.001):
    grad_loss = jax.grad(loss)
    grads = grad_loss(params)
    return params - learning_rate * grads

def optimize_loop(x0, print_loss = False):
    NUM_STEPS = 50000
    for n in range(NUM_STEPS):
        x0 = update_parameters_step(x0)
        if print_loss and n % 10000 == 0:
            print(x0)
    return x0

result = optimize_loop(1.0, print_loss=True)
print(result)