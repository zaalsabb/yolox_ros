import rospy
import numpy as np
from geometry_msgs.msg import Transform
from scipy.optimize import leastsq
from scipy.linalg import svd

def ProjectToImage(projectionMatrix, pos):
    pos = np.array(pos).reshape(3,-1)

    pos_ = np.vstack([pos, np.ones((1, pos.shape[1]))])

    uv_ = np.dot(projectionMatrix, pos_)
    # uv_ = uv_[:, uv_[-1, :] > 0]
    uv = (uv_[:-1, :]/uv_[-1, :]).T

    return uv

def ProjectToWorld(pos1,R,projectionMatrix, uv):
    pos1 = np.array(pos1).reshape(3,1)
    uv = np.array(uv).reshape(2,-1)

    uv_ = np.vstack([uv[0,:], uv[1,:], np.ones((1, uv.shape[1]))])
    pinvProjectionMatrix = np.linalg.pinv(projectionMatrix)

    pos2_ = np.dot(pinvProjectionMatrix, uv_)
    pos2_[-1,pos2_[-1,:]==0] = 1
    pos2 = pos2_[:-1,:]/pos2_[-1,:]

    rays = pos2-pos1
    rays_local = R @ rays
    rays[:,rays_local[2,:]<0]=-1*rays[:,rays_local[2,:]<0]    
    rays = rays/np.linalg.norm(rays,axis=0)

    return rays

def plane_error(p, xs, ys, zs):
    A = p[0]
    B = p[1]
    C = p[2]
    D = p[3]
    return abs(A*xs + B*ys + C*zs + D) / np.sqrt(A**2 + B**2 + C**2)

def plane3point(x,y,z):
    N = len(x)
    L = np.vstack([x,y,z]).T
    d = np.ones((N,1))
    A,B,C = np.linalg.solve(L,d)
    p = np.sum([x,y,z],axis=0)
    D = -1*(A*p[0]+B*p[1]*C*p[2])
    return A,B,C,D

def solve_svd(A,b):
    A_pinv = np.linalg.pinv(A) 
    x = np.dot(A_pinv,b)
    return x

def planeNpoint(points):
    x=points[:,0]
    y=points[:,1]
    z=points[:,2]
    N = len(x)
    L = np.vstack([x,y,z]).T
    d = np.ones((N,1))
    A,B,C = solve_svd(L,d).reshape(-1)
    p = np.sum([x,y,z],axis=0)
    D = -1*(A*p[0]+B*p[1]*C*p[2])
    return A,B,C,D

def plane_leastsq(xs,ys,zs):

    if len(xs) == 3:
        xs = np.insert(xs, 0, np.mean(xs))
        ys = np.insert(ys, 0, np.mean(ys))
        zs = np.insert(zs, 0, np.mean(zs))

    p0 = [1, 1, 1, 1]
    sol = leastsq(plane_error, p0, args=(xs, ys, zs))[0]

    return sol

def PlanePointIntersect(origin,ray,A,B,C,D):
    n = np.array([A, B, C])/np.linalg.norm([A, B, C])
    p0 = np.array([0,0,-D/C])
    d = np.dot(p0 - origin,n)/np.dot(ray,n)
    return origin+d*ray    

def ransac_plane(points,inlier_ratio,max_dist,max_iteration):

    xs=points[:,0]
    ys=points[:,1]
    zs=points[:,2]
    n_data = points.shape[0]
    n_inlier = 0
    iteration = 0    
    ratio = []

    while iteration < max_iteration and n_inlier / n_data < inlier_ratio:
        dist_it=[]
        ind = [np.random.randint(0,n_data) for _ in range(int(n_data/5))]
        xi = np.array([xs[i] for i in ind])
        yi = np.array([ys[i] for i in ind])
        zi = np.array([zs[i] for i in ind])
        try:
            # P = plane3point(xi,yi,zi)
            P = plane_leastsq(xi,yi,zi)
        except:
            continue
        dist_it = plane_error(P, xs, ys, zs)
        index_inliers = np.where(dist_it <= max_dist)
        n_inlier = sum(dist_it <= max_dist)
        ratio.append(n_inlier/n_data)
        iteration +=1
    A,B,C,D = P
    r=ratio[-1]
    return A,B,C,D,r,iteration,index_inliers

def create_transform(p,q):
    
    T = Transform()
    T.translation.x = p[0]
    T.translation.y = p[1]
    T.translation.z = p[2]
    
    T.rotation.x = q[0]
    T.rotation.y = q[1]
    T.rotation.z = q[2]
    T.rotation.w = q[3]
    
    return T

def unpack_transform(T):
    p = [T.translation.x, T.translation.y, T.translation.z]
    q = [T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w]
    return p,q

def unpack_pose(T):
    p = np.array([T.position.x, T.position.y, T.position.z])
    q = np.array([T.orientation.x, T.orientation.y, T.orientation.z, T.orientation.w])
    return p,q    

def points2numpy(pl):
    return np.array([[p.x,p.y,p.z] for p in pl])

def quaterions2numpy(ql):
    return np.array([[q.x,q.y,q.z,q.w] for q in ql])    

def project_2d_to_3d(m,K,D,center=False, h=0):
    v = m[1,:]
    u = m[0,:]
         
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    d = []

    if center:
        d0 = D[int(v.mean()),int(u.mean())]   
        d = [d0 for _ in range(u.shape[0])]  
    else:
        for ui,vi in zip(u,v):
            di = D[int(vi)-h:int(vi)+h+1,int(ui)-h:int(ui)+h+1]
            if len(di)>0:
                di = di[di>0].mean()
                d.append(di)

    Z = np.array(d,dtype=np.float)/1000
    X = Z*(u-cx)/fx
    Y = Z*(v-cy)/fy    

    return X,Y,Z   