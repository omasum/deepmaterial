import torch
import numpy as np

def torch_norm(arr, dim=1, norm=True, keepdims=True):
    length = torch.sqrt(torch.sum(arr * arr, dim = dim, keepdims=keepdims))
    if norm:
        return arr / (length + 1e-12)
    else:
        return length

def numpy_norm(arr, dim=-1):
    length = np.sqrt(np.sum(arr * arr, axis = dim, keepdims=True))
    return arr / (length + 1e-12)

def torch_dot(a,b, dim=-3, keepdims=True):
    return torch.sum(a*b,dim=dim,keepdims=keepdims)

def numpy_dot(a,b):
    return np.sum(a * b, axis=-1)[...,np.newaxis]

def torch_cross(a, b, dim=-3):
    if dim == -3:
        x = a[...,1,:,:] * b[...,2,:,:] - a[...,2,:,:]*b[...,1,:,:]
        y = a[...,2,:,:] * b[...,0,:,:] - a[...,0,:,:]*b[...,2,:,:]
        z = a[...,0,:,:] * b[...,1,:,:] - a[...,1,:,:]*b[...,0,:,:]
    elif dim == -1:
        x = a[...,1] * b[...,2] - a[...,2]*b[...,1]
        y = a[...,2] * b[...,0] - a[...,0]*b[...,2]
        z = a[...,0] * b[...,1] - a[...,1]*b[...,0]
    return torch.stack([x,y,z], dim=dim)

def reflect(d, n, axis=-1):
    # Given a vector of d, calculate the reflect vector coresponding to n.
    # d and n have same original point.
    return 2*n*(np.sum(d*n,axis=axis))-d
