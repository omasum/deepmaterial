import numpy as np
from .vector_util import numpy_norm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
# 抛物面方程 z = a * (x^2 + y^2)
# 焦点计算公式 (0, 0, -1/4a)

# 抛物面镜配置参数：焦点长度f，自变量x范围，旋转矩阵R，平移矩阵T
class paraMirror:
    def __init__(self, opt):
        '''
            f: The focal length of parabolic mirror. The function of mirror is formed by rotated y=a*x^2 around z axis.
            xy_range: The area of parabolic section, given the range of x and y.
            R: The rotation matrix translating the world coordinate system to parabolic mirror coordinate system. 3x3 matrix
            T: The translation matrix translating the world coordinate system to parabolic mirror coordinate system. 1x3 vector
            xy_reso: the xy_range should be descrete into how many pixels 
        '''
        # *There is three coordinate systems which are local, world, mirror coordinate system repectively.
        # *The coordinate system of direction is marked in the bracket.
        f = opt['f']
        dia = opt.get('dia', 0)
        yoff = opt.get('y_offset', 0)
        self.yoff = yoff
        self.dia = dia
        if dia != 0 and yoff != 0:
            xy_range = {
                'x':[-dia/2,dia/2],
                'y':[yoff-dia/2,yoff+dia/2]
            }
        else:
            xy_range = opt['xy_range']
        R = Rotation.from_euler(opt['rotation_mode'],opt['eular'], degrees=True)
        R = R.as_matrix()
        T = np.array(opt['T'])
        
        self.f = np.array([0,0,f],dtype=np.float32)
        self.focallength = f
        self.a = 1/(4*f)
        self.R = R
        self.T = T
        if opt.get('fixed_light', False):
            self.o = opt['o']
            self.d = opt['d']
            self.fixed_light = True
        else:
            self.fixed_light = False
        self.xreso = opt['xy_reso']['x']
        self.yreso = opt['xy_reso']['y']
        self.setXYrange(xy_range=xy_range)
            
    
    def func(self, x,y):
        a = 1/(4*self.focallength)
        return a*(x**2+y**2)
    def get_rangen(self):
        # With the known parameters of mirror, calculate the range of normal in which the measurements is maintained in upper-hemisphere of the sample point.
        
        return

    def intersect(self, o, d):
        # Given a ray, retrun the intersect point of ray and parabolic function
        o = self.__w2m(o)
        d = self.__w2m(d)
        alpha, beta, gamma = d
        a = b = self.a
        c = 0
        x1, y1, z1 = o
        p = gamma**2-4*a*alpha**2*c-4*b*beta**2*c\
            +4*a*alpha**2*z1 + 4*b*beta**2*z1\
            -4*a*alpha*gamma*x1 - 4*b*beta*gamma*y1\
            -4*a*b*beta**2*x1**2 - 4*a*alpha**2*b*y1**2+8*a*b*alpha*beta*x1*y1;
        x = []
        t1 = None
        t2 = None # 这里随便给了t的初始值
        if alpha == 0 and beta == 0:
            z = self.func(x1, y1)
            t1 = -(z-z1)
        else:
            if(p==0): # 根号里的内容为0，说明t有一个解，一个交点
                sig = np.sqrt(p);
                t1 =  -(sig-gamma+2*a*alpha*x1+2*b*beta*y1)/(2*(a*alpha**2+b*beta**2))
            elif(p>0): #根号里的内容大于0，说明t有两个解，两个交点
                sig = np.sqrt(p);
                t1 =  -(sig-gamma+2*a*alpha*x1+2*b*beta*y1)/(2*(a*alpha**2+b*beta**2))
                t2 =  -(-sig-gamma+2*a*alpha*x1+2*b*beta*y1)/(2*(a*alpha**2+b*beta**2))
        if t1 is not None:
            x.append(self.__m2w(o+t1*d))
        if t2 is not None:
            x.append(self.__m2w(o+t2*d))
        return x

    def idx2xy(self, xidx, yidx):
        x = (xidx/self.xreso)*(self.xr[1]-self.xr[0]) + self.xr[0]
        y = (yidx/self.yreso)*(self.yr[1]-self.yr[0]) + self.yr[0]
        return x, y
    def in_mirror(self, x, y):
        status = ((x-self.center[0])**2 + (y-self.center[1])**2) < (self.dia/2)**2
        return status
    def sample_dir(self, world):
        # sample a point on mirror, and calculate the normalized direction from focus to this point
        x,y = self.xr[0], self.yr[0]
        while not self.in_mirror(x,y):
            x = np.random.uniform(self.xr[0], self.xr[1])
            y = np.random.uniform(self.yr[0], self.yr[1])
        z = self.func(x,y)
        p = np.array([x,y,z], dtype=np.float32)
        d = numpy_norm(p-self.f)
        if world:
            d = self.__m2w(d)
        return d

    def setXYrange(self, xy_range):
        self.xr = xy_range['x']
        self.yr = xy_range['y']
        x = np.arange(self.xr[0], self.xr[1], (self.xr[1]-self.xr[0])/self.xreso)
        y = np.arange(self.yr[0], self.yr[1], (self.yr[1]-self.yr[0])/self.yreso)
        x_mat, y_mat = np.meshgrid(x,y)
        z = self.func(x_mat,y_mat)
        self.mask = self.get_mask(x_mat, y_mat)
        self.xyz = np.stack([x_mat,y_mat,z],axis=0)
        self.center = [(self.xr[0]+self.xr[1])/2, (self.yr[0]+self.yr[1])/2]

    def get_mask(self, xmat, ymat):
        # With known y offset and diameter of the mirror, return the mask in camera pixel
        center = [(self.xr[0]+self.xr[1])/2, (self.yr[0]+self.yr[1])/2]
        mask = ((xmat-center[0])**2 + (ymat-center[1])**2) < (self.dia/2)**2
        return mask
    def focuse_ray(self, x, y):
        # Given a direction (world) paralleled to z axis (mirror), calculate the reflected direction (world). d is a 1x3 vector
        # md = self.__w2m(d)
        z = self.func(x,y)
        mref = numpy_norm(np.array([x,y,z],dtype=np.float32)-self.f)
        return self.__m2w(mref)

    def __w2m(self, d):
        # Convert the direction from world coordinate system to mirror coordinate system. d is a 1x3 vector
        # if len(d.shape) == 1:
        #     np.expand_dims(d,axis=0)
        wd = np.matmul(self.R,d)+self.T
        # if len(d.shape) == 1:
        #     wd = np.squeeze(wd, axis=0)
        return wd

    def __m2w(self, d):
        # Convert the direction from mirror coordinate system to world coordinate system. d is a 1x3 vector
        if len(d.shape) == 3:
            vec = d.reshape(3,-1)-self.T.reshape(3,1)
        else:
            vec = d-self.T
        result = np.matmul(self.R.T, vec)
        if len(d.shape) == 3:
            result = result.reshape(3,*d.shape[-2:])
        return result

    def get_wo(self, TBN=None, world=False, jitter=False, f = None):
        # Get the wo matrix (local), given the surface normal direction (world). 
        # In local coordinate system, n is z-axis, t is x-axis, b is y-axis. t,b,n is 1x3 vectors
        if f is None:
            # Following code (if jitter) attempt to consider the noise brought by the focus and 3D point alignment. 
            # However, this code only simulate the noise by jitter the position of focus. 
            # While the out lighting is no more paralleled to the optical axis.
            f = self.f if not jitter else self.f+np.random.normal(loc=0,scale=1e-2,size=(3,))
        # x = np.arange(self.xr[0], self.xr[1], (self.xr[1]-self.xr[0])/self.xreso)
        # y = np.arange(self.yr[0], self.yr[1], (self.yr[1]-self.yr[0])/self.yreso)
        # x_mat, y_mat = np.meshgrid(x,y)
        # z = self.func(x_mat,y_mat)
        xyz = self.xyz
        md = numpy_norm(xyz-f.reshape(3,1,1), dim=0)
        wo = self.__m2w(md)
        if not world:
            wo = np.matmul(TBN, wo.reshape(3,-1)).reshape(3,*wo.shape[-2:])
        theta = np.arccos(np.minimum(wo[2,:,:], 1.0))
        phi = np.arctan(wo[1,:,:]/wo[0,:,:])
        phi = (wo[0,:,:] < 0) * (phi + np.pi) + (wo[0,:,:] > 0) * phi
        s = np.stack([theta, phi], axis=0)
        mask = self.mask
        if not jitter:
            return wo, s, mask
        else:
            return wo, f, s, mask
    
    def plt_func(self):
                
        #numpy.meshgrid()→生成网格点坐标矩阵
        x = np.arange(self.xr[0], self.xr[1], (self.xr[1]-self.xr[0])/self.xreso)
        y = np.arange(self.yr[0], self.yr[1], (self.yr[1]-self.yr[0])/self.yreso)
        x_mat, y_mat = np.meshgrid(x,y)
        mask = self.get_mask(xmat=x_mat, ymat=y_mat)
        z = self.func(x_mat,y_mat) * mask
        fig = plt.figure()
        ax = plt.axes(projection= "3d")
        #ax.contour3D(x,y,z,5,cmap='binary')#绘制成等高线图
        ax.plot_surface(x,y,z,rstride = 1,cstride = 1,cmap = 'viridis',edgecolor = 'none')
        ax.scatter(0, 0, self.f, c='g', label='green points')

        #设置标签
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def get_wi(self, o=None, d=None, TBN=None, world=False, f=None, jitter=False):
        # Get the wi direction (local), given the (x,y) (mirror) of the incident direction 
        # which is paralleled to z axis in mirror coordinate system.
        if self.fixed_light:
            o = self.o
            d = self.d
        else:
            if o is None or d is None:
                raise ValueError("light origin and direction is required!")
        point = self.intersect(o,d)[0]
        if f is None:
            f = self.f if not jitter else self.f+np.random.normal(loc=0,scale=1e-2,size=(3,))
        
        mpoint = self.__w2m(point)
        mwi = numpy_norm(mpoint-f)
        wwi = self.__m2w(mwi)
        if world:
            return wwi
        else:
            lwi = np.matmul(TBN, wwi)
            return lwi