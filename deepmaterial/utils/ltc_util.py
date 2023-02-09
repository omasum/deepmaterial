import numpy as np
import torch
# from deepmaterial.utils.render_util import Directions, RectLighting

from deepmaterial.utils.vector_util import torch_cross, torch_dot, torch_norm, numpy_norm
from deepmaterial.utils.wrapper_util import timmer

class LTC:
    def __init__(self, X=[1,0,0], Y=[0,1,0], Z=[0,0,1], amplitude=1.0, m00=1.0, m11=1.0, m02=0.0, m20=0.0, device='cpu', MIN_ALPHA=0.0001, mode='fitting'):
        self.m00 = m00
        self.m11 = m11
        self.m02 = m02
        self.m20 = m20
        self.amplitude = amplitude
        self.X = numpy_norm(np.array(X, dtype=np.float32), dim=0)
        self.Y = numpy_norm(np.array(Y, dtype=np.float32), dim=0)
        self.Z = numpy_norm(np.array(Z, dtype=np.float32), dim=0)
        self.device = device
        self.MIN_ALPHA=MIN_ALPHA
        self.init(mode)
    
    def sampleClampedCos(self, u, v):
        cosTheta = torch.sqrt(u)
        sinTheta = torch.sqrt(1-cosTheta ** 2)
        phi = 2 * torch.pi * v
        vec = torch.stack([sinTheta * torch.cos(phi), sinTheta * torch.sin(phi), cosTheta], dim=-1).unsqueeze(-1)

        return vec
    
    def sample(self, u, v):
        vec = self.sampleClampedCos(u,v)
        
        #* translate the vector sampled from cosine distribution to the distribution of we want
        vec = torch_norm(torch.matmul(self.M, vec), dim=-2).permute(3, 2, 0, 1).contiguous()
        return vec
    
    def init(self, mode):
        if mode == 'fitting':
            self.rotMat = torch.stack([torch.from_numpy(self.X), torch.from_numpy(self.Y), torch.from_numpy(self.Z)], dim=1).to(self.device)
            self.scaleMat = torch.from_numpy(np.array([[self.m00, 0.0, self.m02], [0.0, self.m11, 0.0], [self.m20, 0.0, 1.0]], dtype=np.float32)).to(self.device)
        self.update(self.m00, self.m11, self.m02, self.m20, mode=mode)
    
    def update(self, m00, m11, m02, m20, mode='fitting'):
        '''Update the parameters of ltc
        '''
        if mode=='fitting':
            self.M = torch.matmul(self.rotMat, self.scaleMat)
            self.m00 = np.maximum(self.MIN_ALPHA, m00)
            self.scaleMat[0,0] = self.m00
            self.m11 = np.maximum(self.MIN_ALPHA, m11)
            self.scaleMat[1,1] = self.m11
            self.m02 = m02
            self.scaleMat[0,2] = m02
            self.m20 = m20
            self.scaleMat[2,0] = m20
        
            self.invM = torch.linalg.inv(self.M)
            self.det = torch.abs(torch.det(self.invM))
        else:
            # paddingOnes = torch.ones_like(self.m00)
            # paddingZeros = torch.zeros_like(self.m00)
            self.invM = torch.eye(3, device=self.m00.device).view((1,)*self.m00.ndim + (3,3)).repeat(*(self.m00.shape + (1,1)))
            self.invM[...,2,2] = self.m00
            self.invM[...,1,1] = (m00 - m02*m20)/m11
            self.invM[...,2,0] = -m20
            self.invM[...,0,2] = -m02
            self.invM.squeeze_(1)

    def DoClampedCos(self, vec, n):
        res = torch_dot(vec, n, dim=-3)
        res = torch.clip(res, 0.0, 1.0)/torch.pi
        return res
    
    def eval(self, w, n):
        wo = torch.matmul(self.invM, w.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1)
        length = torch.sqrt((wo**2).sum(-1))
        jacobian = self.det / length**3
        
        wo = torch_norm(wo, dim=-1).permute(0, 3, 1, 2).contiguous()
        a = self.DoClampedCos(wo, n)
        
        return a * jacobian * self.amplitude

    def norm(self, nSample=1024):
        stepu = torch.pi / nSample
        stepv = 2*torch.pi / nSample
        u = torch.arange(0, torch.pi, stepu)
        v = torch.arange(0, 2*torch.pi, stepv)
        u, v = torch.meshgrid(u, v, indexing='ij')
        n = torch.stack([torch.zeros_like(u), torch.zeros_like(u), torch.ones_like(u)], dim=0).unsqueeze(0)
        
        sinTheta = torch.sin(u)
        vec = torch.stack([sinTheta*torch.cos(v), sinTheta*torch.sin(v), torch.cos(u)], dim=0).unsqueeze(0)
        res = self.eval(vec, n)

        #* spherical integration of the transformed distribution
        norm = (res * stepu * stepv * sinTheta).sum((0,2,3))
        print("Norm of the distribution: ", norm)
        return stepu, stepv, vec, res, sinTheta

    def integrateEdge(self, v1, v2, dim=-3):
        cosTheta = torch.clip(torch_dot(v1, v2, dim=dim),-1.0+1e-6, 1.0-1e-6)
        theta = torch.acos(cosTheta)

        coefficiency = theta/torch.sin(theta)
        res = torch_cross(v1, v2, dim=dim) * (torch.where((theta > 0.001), coefficiency, torch.ones_like(coefficiency)))
        return res

    def evalPolygon(self, vec, s=0.0, fresNorm=0.0, isDiff=False):
        if isDiff:
            l = vec.permute(0, 4, 3, 1, 2).contiguous()
        else:
            l = torch.matmul(self.invM, vec).permute(0, 4, 3, 1, 2).contiguous()
        #* transform the lighting direction (un-normalized) to cosine distribution
        lClipped, n = LTC.clipPolygon(l, normalization=True)
        vLookUp = 0.0
        #! 当矩形所有点被裁剪后，得到的向量为0向量，需要特殊处理
        for i in range(lClipped.shape[-4]-1):
            vLookUp = vLookUp + self.integrateEdge(lClipped[:, i], lClipped[:, i+1]) * torch.clip((n-i), 0, 1).unsqueeze(1)
        res = vLookUp[:, 2:3].clip_(0.0)

        if not isDiff:
            res = (s*self.amplitude + (1-s)*fresNorm) * res

        #* 1/(2*pi) is the factor in the fomular of integrating polygon
        return res / 2 / torch.pi, l, vLookUp, n

    def testSampling(self, nSample=1024):
        stepu, stepv, vec, res, sinTheta = self.norm(nSample=nSample)
        
        #* ground truth moment
        momentGT = (res * vec * stepu * stepv * sinTheta).sum((0,2,3))
        print("ground truth moment: ", momentGT)

        #* important sampling moment
        u = (torch.arange(0, nSample, 1, dtype=torch.float32) + 0.5)/nSample
        v = (torch.arange(0, nSample, 1, dtype=torch.float32) + 0.5)/nSample
        u, v = torch.meshgrid(u, v, indexing='ij')

        vec = self.sample(u,v)
        momentEs = vec.mean((0,2,3))
        print("Important sampling moment: ", momentEs)
    
    def testIntegratePolygon(self, vec):
        n = torch_norm(torch.cross(vec[0]-vec[1], vec[0]-vec[2]), dim=0).view(3,1,1)
        A = torch.norm(torch.cross(vec[0]-vec[1], vec[0]-vec[2]))/2

        res = 0.0
        dx = 0.0025
        u = torch.arange(0, 1, dx, dtype=torch.float32)
        v = torch.arange(0, 1, dx, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='ij')
        nplane = torch.from_numpy(np.array([0, 0, 1], dtype=np.float32).reshape(1, 3, 1, 1))
        p = torch.where(u>v, (1-u).unsqueeze(0)*vec[0].view(3,1,1)+(u-v).unsqueeze(0)*vec[1].view(3,1,1)+v.unsqueeze(0)*vec[2].view(3,1,1)\
            , (1-v).unsqueeze(0)*vec[0].view(3,1,1)+(v-u).unsqueeze(0)*vec[1].view(3,1,1)+u.unsqueeze(0)*vec[2].view(3,1,1))

        w = torch_norm(p, dim=0)
        d2 = torch.norm(p, dim=0, keepdim=True) ** 2
        res = dx * dx * A * torch.abs(torch_dot(w, n, dim=0)) / d2 * self.eval(w.unsqueeze(0), nplane)
        print("polygon integration of D is: ", res.sum())

        vec = (self.invM @ vec.T).T
        n = torch_norm(torch.cross(vec[0]-vec[1], vec[0]-vec[2]), dim=0).view(3,1,1)
        A = torch.norm(torch.cross(vec[0]-vec[1], vec[0]-vec[2]))/2
        dx = 0.0025
        u = torch.arange(0, 1, dx, dtype=torch.float32)
        v = torch.arange(0, 1, dx, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='ij')
        nplane = torch.from_numpy(np.array([0, 0, 1], dtype=np.float32).reshape(1, 3, 1, 1))
        p = torch.where(u>v, (1-u).unsqueeze(0)*vec[0].view(3,1,1)+(u-v).unsqueeze(0)*vec[1].view(3,1,1)+v.unsqueeze(0)*vec[2].view(3,1,1)\
            , (1-v).unsqueeze(0)*vec[0].view(3,1,1)+(v-u).unsqueeze(0)*vec[1].view(3,1,1)+u.unsqueeze(0)*vec[2].view(3,1,1))

        w = torch_norm(p, dim=0)
        d2 = torch.norm(p, dim=0, keepdim=True) ** 2
        res = dx * dx * A * torch.abs(torch_dot(w, n, dim=0)) / d2 * self.DoClampedCos(w.unsqueeze(0), nplane)
        print("polygon integration of Do is: ", res.sum())

    @staticmethod
    def clipPolygon(polygon, normalization=True):
        #TODO 适配数据结构，并确定其可微与否是否有影响
        #* detect clipping config
        b, nl, c, h, w = polygon.shape
        n = torch.zeros((b,h,w), dtype=torch.int16).to(polygon.device)
        config = torch.zeros((b,h,w), dtype=torch.int32).to(polygon.device)
        consOne = torch.ones_like(config)
        consZero = torch.zeros_like(config)
        config += torch.where(polygon[:, 0, 2] > 0.0, consOne, consZero)
        config += torch.where(polygon[:, 1, 2] > 0.0, consOne*2, consZero)
        config += torch.where(polygon[:, 2, 2] > 0.0, consOne*4, consZero)
        config += torch.where(polygon[:, 3, 2] > 0.0, consOne*8, consZero)

        #* count the final vertex count of each pixel
        consOne = torch.ones((b,h,w), dtype=torch.int16).to(polygon.device)
        n = torch.where((config==1)+(config==2)+(config==4)+(config==8), consOne*3, n)
        n = torch.where((config==3)+(config==6)+(config==9)+(config==12)+(config==15), consOne*4, n)
        n = torch.where((config==7)+(config==11)+(config==13)+(config==14), consOne*5, n)

        #* expand a vertex for clip
        polygon = torch.cat([polygon, polygon[:, 0:1]], dim=1)
        polygonClipped = polygon.clone()

        # if normalization:
        #     polygon = torch_norm(polygon, dim=1)
        # return polygon, n
        config = config.view(b,1,1,h,w)
        #* clip
        #* V1 clip V2 V3 V4
        polygonClipped[:, 1:3] = torch.where(config==1, \
            torch.stack([-polygon[:, 1,2:3]*polygon[:, 0]+polygon[:, 0,2:3]*polygon[:, 1],-polygon[:, 3,2:3]*polygon[:, 0]+polygon[:, 0,2:3]*polygon[:, 3]], dim=1), polygonClipped[:, 1:3])
        #* V2 clip V1 V3 V4
        polygonClipped[:, 0:3:2] = torch.where(config==2, \
            torch.stack([-polygon[:, 0,2:3]*polygon[:, 1]+polygon[:, 1,2:3]*polygon[:, 0],-polygon[:, 2,2:3]*polygon[:, 1]+polygon[:, 1,2:3]*polygon[:, 2]], dim=1), polygonClipped[:, 0:3:2])
        #* V1 V2 clip V3 V4
        polygonClipped[:, 2:4] = torch.where(config==3, \
            torch.stack([-polygon[:, 2,2:3]*polygon[:, 1]+polygon[:, 1,2:3]*polygon[:, 2],-polygon[:, 3,2:3]*polygon[:, 0]+polygon[:, 0,2:3]*polygon[:, 3]], dim=1), polygonClipped[:, 2:4])
        #* V3 clip V1 V2 V4
        polygonClipped[:, 0:2] = torch.where(config==4, \
            torch.stack([-polygon[:, 3,2:3]*polygon[:, 2]+polygon[:, 2,2:3]*polygon[:, 3],-polygon[:, 1,2:3]*polygon[:, 2]+polygon[:, 2,2:3]*polygon[:, 1]], dim=1), polygonClipped[:, 0:2])
        #* V2 V3 clip V1 V4
        polygonClipped[:, ::3] = torch.where(config==6, \
            torch.stack([-polygon[:, 0,2:3]*polygon[:, 1]+polygon[:, 1,2:3]*polygon[:, 0],-polygon[:, 3,2:3]*polygon[:, 2]+polygon[:, 2,2:3]*polygon[:, 3]], dim=1), polygonClipped[:, ::3])
        #* V1 V2 V3 clip V4
        polygonClipped[:, 3:5] = torch.where(config==7, \
            torch.stack([-polygon[:, 3,2:3]*polygon[:, 2]+polygon[:, 2,2:3]*polygon[:, 3],-polygon[:, 3,2:3]*polygon[:, 0]+polygon[:, 0,2:3]*polygon[:, 3]], dim=1), polygonClipped[:, 3:5])
        #* V4 clip V1 V2 V3
        polygonClipped[:, 0:3] = torch.where(config==8, \
            torch.stack([-polygon[:, 0,2:3]*polygon[:, 3]+polygon[:, 3,2:3]*polygon[:, 0],-polygon[:, 2,2:3]*polygon[:, 3]+polygon[:, 3,2:3]*polygon[:, 2],polygon[:, 3]], dim=1), polygonClipped[:, 0:3])
        #* V1 V4 clip V2 V3
        polygonClipped[:, 1:3] = torch.where(config==9, \
            torch.stack([-polygon[:, 1,2:3]*polygon[:, 0]+polygon[:, 0,2:3]*polygon[:, 1],-polygon[:, 2,2:3]*polygon[:, 3]+polygon[:, 3,2:3]*polygon[:, 2]], dim=1), polygonClipped[:, 1:3])
        #* V1 V2 V4 clip V3
        polygonClipped[:, 2:5] = torch.where(config==11, \
            torch.stack([-polygon[:, 2,2:3]*polygon[:, 1]+polygon[:, 1,2:3]*polygon[:, 2],-polygon[:, 2,2:3]*polygon[:, 3]+polygon[:, 3,2:3]*polygon[:, 2],polygon[:, 3]], dim=1), polygonClipped[:, 2:5])
        #* V3 V4 clip V1 V2
        polygonClipped[:, :2] = torch.where(config==12, \
            torch.stack([-polygon[:, 0,2:3]*polygon[:, 3]+polygon[:, 3,2:3]*polygon[:, 0],-polygon[:, 1,2:3]*polygon[:, 2]+polygon[:, 2,2:3]*polygon[:, 1]], dim=1), polygonClipped[:, :2])
        #* V1 V3 V4 clip V2
        polygonClipped[:, 1:5] = torch.where(config==13, \
            torch.stack([-polygon[:, 1,2:3]*polygon[:, 0]+polygon[:, 0,2:3]*polygon[:, 1],-polygon[:, 1,2:3]*polygon[:, 2]+polygon[:, 2,2:3]*polygon[:, 1], polygon[:, 2], polygon[:, 3]], dim=1), polygonClipped[:, 1:5])
        #* V2 V3 V4 clip V1
        polygonClipped[:, ::4] = torch.where(config==14, \
            torch.stack([-polygon[:, 0,2:3]*polygon[:, 1]+polygon[:, 1,2:3]*polygon[:, 0],-polygon[:, 0,2:3]*polygon[:, 3]+polygon[:, 3,2:3]*polygon[:, 0]], dim=1), polygonClipped[:, ::4])
        
        config = config.view(b,1,h,w)

        polygonClipped[:, 3] = torch.where(n.view(b,1,h,w)==3, polygonClipped[:, 0], polygonClipped[:, 3])
        polygonClipped[:, 4] = torch.where(n.view(b,1,h,w)==4, polygonClipped[:, 0], polygonClipped[:, 4])
        if (n==5).any():
            polygonClipped = torch.cat([polygonClipped, polygonClipped[:, 0:1]], dim=1)
        if normalization:
            polygonClipped = torch_norm(polygonClipped, dim=-3)
        return polygonClipped, n


if __name__ == '__main__':
    lut = torch.load('deepmaterial/utils/LTC/look-up-table.pth')
    r = -0.6
    theta = 60
    nSample=64
    t = round(theta / 90 * (nSample-1))
    a = round((r/2+0.5) * (nSample-1))
    #* indexing the look-up-table and fetch the parameters
    params = lut[a, t]
    ltc = LTC(m02=1)
    ltc.M = torch.from_numpy(np.array([[0.1, 0.5, 1], [0.2, 1, 1], [0, 2, 0.5]], dtype=np.float32))
    ltc.invM = torch.linalg.inv(ltc.M)
    ltc.det = torch.linalg.det(ltc.invM)
    ltc.amplitude = 1.0
    ltc.testSampling()
    vec = torch.from_numpy(np.array([[0,0,1], [1,0,1], [1,1,1]], dtype=np.float32))
    ltc.testIntegratePolygon(vec)