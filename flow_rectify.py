from doctest import SKIP
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from util import *

class RectifyFlowOfMoving(object):
    def __init__(self,T_4x,K_cam,n,A,moving_probs,flow,occlusions,ground,sampling,sampling_for_ground):
        self.T_4x = T_4x
        self.R_cam,self.T_cam = R_t(T_4x[0])
        self.K_cam = K_cam
        self.n = n
        self.moving_probs = moving_probs
        self.flow = flow
        self.occlusions = occlusions
        self.ground = ground
        self.sampling = sampling
        self.samp_ground = sampling_for_ground
        self.A = A
        #self.inter_depth = inter_depth

    def on_batch(self):
        moving_probs = self.moving_probs
        ## recompute the optical flow for moving objects
        flow_rec = self.flow.copy()
        for i in range(len(moving_probs)):
            p1,p2,R,T,moving_probs[i],flag = self.get_R_T(self.flow,moving_probs[i],self.sampling,self.K_cam)
            if len(p1) == 0:
                continue
            elif flag == False:
                continue
            else:
                _3d_pts = self.get_3d_points(R, T, self.K_cam, p1.copy(), p2.copy())
                #reproj_error = self.get_reproj_error(_3d_pts,R_opt,T_opt,self.K_cam,p2.copy())
                #print(reproj_error)
                p1_ground,p2_ground = self.get_features_for_ground(self.flow,self.occlusions,self.ground, self.samp_ground)
                _3d_pts_ground = self.get_3d_points(self.R_cam, self.T_cam, self.K_cam, p1_ground, p2_ground)
                s_obj,s_ground = self.estimate_scale(_3d_pts_ground,_3d_pts,self.n)
                p2_reproj = self.reproj( _3d_pts, self.R_cam, self.T_cam, self.K_cam, s_obj)
                #flow_rec[moving_probs[i]==1,:2] = 0
                p1 = p1.astype('int')
                flow_rec[p1.T[1],p1.T[0],:2] = p2_reproj - p1
                #flow_rec = self.cal_flow_rec(flow_rec,moving_probs[i],p1,p2_reproj)
                #flow_rec[p1.T[1],p1.T[0],:2] = p2_reproj - p1
        g = self.cal_structure(flow_rec,self.K_cam, self.A, self.T_4x)
        #g_gt = self.cal_structure_gt(self.flow,self.inter_depth,self.R_cam,self.T_cam,self.K_cam,s_ground,self.A,self.T_4x,self.n)
        return flow_rec,g,moving_probs#,g_gt

    def get_3d_points(self, R_obj, T_obj, K, p1, p2):
        R_init = np.eye(3, 3)
        T_init = np.zeros((3, 1))
        ## construct projection matrix
        M1 = np.zeros((3, 4))
        M2 = np.zeros((3, 4))
        M1[0:3, 0:3] = R_init
        M1[:, 3] = T_init.T
        M2[0:3, 0:3] = R_obj
        M2[:, 3] = T_obj.T
        M1 = np.dot(K, M1)
        M2 = np.dot(K, M2)
        
        ## triangulate to get 3D points
        _3D_homo = cv2.triangulatePoints(M1, M2, p1.T, p2.T)
        _3D = _3D_homo[:3,:]/_3D_homo[3,:]

        ## bundle adjustment
        _3d_pts = self.bundle_adjustment( R_init, T_init, K, _3D, p1)

        return _3d_pts
    
    def estimate_scale(self,_3d_pts_ground,_3d_pts,n):

        s_ground = 1/n.dot(_3d_pts_ground.T)
        up_bound = np.percentile(s_ground,95)
        low_bound = np.percentile(s_ground,5)
        s_ground[s_ground<low_bound] = 0
        s_ground[s_ground>up_bound] = 0
        s_ground = s_ground.sum()/len(np.where(s_ground!=0)[0])
        #s_ground = s_ground.mean()

        s_obj = 1/n.dot(_3d_pts.T) 
        cut_pos = int(len(_3d_pts)/5)*4
        #print(cut_pos)
#         if cut_pos/4 > 5000:
#             s_obj = np.percentile(s_obj[len(_3d_pts)-5000:], 5)
#         else:
        s_obj = np.percentile(s_obj[cut_pos:], 5)
        s_obj = s_obj / s_ground
        #print(s_obj)
        return s_obj,s_ground

    def reproj(self, _3d_pts, R_cam, T_cam, K_cam, s):

        ## reprojection 3d points to image
        p2_reproj,J = cv2.projectPoints(_3d_pts*s, R_cam, T_cam, K_cam, np.array([]))
        
        return p2_reproj[:,0,:]
    
    def get_features_from_flow(self,flow,threshold,sampling):

        # Sample features from flow maps
        featurelocations = np.ones(flow.shape[:2]).astype('bool')
        # featurelocations[::sampling,::sampling] = True
        
        # Filter out locations that overlap with wanted areas
        featurelocations = np.logical_and(featurelocations, threshold)

        y,x = np.mgrid[:featurelocations.shape[0],:featurelocations.shape[1]]
        x_feats = x[featurelocations]
        y_feats = y[featurelocations]

        features1 = np.c_[x_feats,y_feats]

        matches_pairwise = []
    
        u = flow[:,:,0]
        v = flow[:,:,1]
        # Get feature motion according to flow
        uloc = u[features1[:,1],features1[:,0]]
        vloc = v[features1[:,1],features1[:,0]]

        uvloc = np.c_[uloc,vloc]
        features2 = features1 + uvloc
        
        features1_currentframe = features1
        features2_currentframe = features2


        matches_pairwise.append(np.dstack((features1_currentframe,features2_currentframe)))

        # for i in range(len(matches_pairwise)):
        #     print('Matches in frame {}: {}'.format(i,matches_pairwise[i].shape[0]))

        return matches_pairwise

    def get_R_T(self, flow, prob, sampling, K_cam):
        flag = True
        matches_pairwise= self.get_features_from_flow(flow,prob,sampling)
        p = matches_pairwise[0].reshape(matches_pairwise[0].shape[0],4)
        p1 = p[:,[0,2]]
        p2 = p[:,[1,3]]
        #print('########object#######')
        init_num = len(p1)
        print(len(p1))
        E,mask = cv2.findEssentialMat (p1, p2, K_cam, method=cv2.RANSAC)
        outlier = p1[np.where(mask==0)[0]]
        outlier_num = len(outlier)
        if outlier_num/init_num > 0.2:
            outlier = outlier.astype('int')
            prob[outlier.T[1],outlier.T[0]] = 0
        p1 = p1[np.where(mask!=0)[0],:]
        p2 = p2[np.where(mask!=0)[0],:]
        _, R, T, mask = cv2.recoverPose(E, p1, p2, K_cam, mask)
        ## filter outlier 
        p1 = p1[np.where(mask!=0)[0],:]
        p2 = p2[np.where(mask!=0)[0],:]
        print('Number of good matches for moving object: {}'.format(len(p1)))
        if len(p1)/init_num < 0.5:
            print('R,T estimation failed.')
            flag = False
        return p1,p2,R,T,prob,flag
    
    def get_features_for_ground(self,flow, occlusions, ground, sampling):
        """
        This functions extract features from the flow. Adapted from generate_features_from_flow function of mrflow.

        Additionally, it does two pre-filtering steps to remove 
        outliers and severely non-rigid points:
        1) A simple removal of the largest features is performed, and
        2) A fundamental matrix F is estimated robustly. The outliers
        in this step are taken to be outliers in the true motion as well.

        """

        # Sample features from flow maps
        featurelocations = np.zeros(flow.shape[:2]).astype('bool')
        featurelocations[::sampling,::sampling] = True

        # Filter out features that are in occlusions / uncertain areas
        occlusions_bwd, occlusions_fwd = occlusions
        occlusion_fwd_or_bwd = np.logical_or(occlusions_bwd>0, occlusions_fwd>0)
        featurelocations[occlusion_fwd_or_bwd] = False

        # Filter out locations that overlap with rigid areas
        featurelocations = np.logical_and(featurelocations, ground)

        y,x = np.mgrid[:featurelocations.shape[0],:featurelocations.shape[1]]
        x_feats = x[featurelocations]
        y_feats = y[featurelocations]

        features1 = np.c_[x_feats,y_feats]

        matches_pairwise = []

        u = flow[:,:,0]
        v = flow[:,:,1]
        # Get feature motion according to flow
        uloc = u[features1[:,1],features1[:,0]]
        vloc = v[features1[:,1],features1[:,0]]

        uvloc = np.c_[uloc,vloc]
        features2 = features1 + uvloc
        #
        # Filter step 1: Remove too large features
        #
        featurelengths = np.sqrt((uvloc**2).sum(axis=1))
        data = np.c_[features1,featurelengths].astype('float64')
        data /= data.std(axis=0)

        inliers_largefilter = np.ones(features1.shape[0]) > 0

        # Remove too large features from set
        features1_currentframe = features1[inliers_largefilter,:]
        features2_currentframe = features2[inliers_largefilter,:]

        #
        # Filter step 2: Remove features according to homography
        #
        F,inliers_features = cv2.findFundamentalMat(features1_currentframe.astype('float32'), features2_currentframe.astype('float32'), method=cv2.LMEDS)
        inliers_features = inliers_features.ravel()>0
        #print('FundmatFilter: Removing {} of {} features.'.format((inliers_features==0).sum(),inliers_features.size))
        rel_retain = inliers_features.astype('float').sum() / inliers_features.size

        if rel_retain < 0.5:
            print('(WW) FundmatFilter would remove more than 50%, skipping...')
            inliers_features[:] = True

        # And again, save outliers
        features1_currentframe = features1_currentframe[inliers_features==True, :]
        features2_currentframe = features2_currentframe[inliers_features==True, :]


        matches_pairwise.append(np.dstack((features1_currentframe,features2_currentframe)))

        # for i in range(len(matches_pairwise)):
        #     print('Matches for ground: {}'.format(matches_pairwise[i].shape[0]))

        p = matches_pairwise[0].reshape(matches_pairwise[0].shape[0],4)
        p1 = p[:,[0,2]]
        p2 = p[:,[1,3]]

        return p1,p2

    def normalize_data(self,x):
        x_mean = np.mean(x)
        x_std = np.std(x,ddof=1)
        x_norm = (x-x_mean)/x_std
        return x_mean,x_std,x_norm

    def linear_estimation(self,p1,p2_reproj):
        ## using Least Squares Method
        A = np.ones((p1.shape[0],3))
        A0_mean,A0_std,A[:,0] = self.normalize_data(p1.T[0])
        A1_mean,A1_std,A[:,1] = self.normalize_data(p1.T[1])
        bx_mean,bx_std,b_x = self.normalize_data((p2_reproj - p1).T[0])
        by_mean,by_std,b_y = self.normalize_data((p2_reproj - p1).T[1])
        coef_x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b_x)
        coef_y = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b_y)
        return A0_mean,A0_std,A1_mean,A1_std,bx_mean,bx_std,by_mean,by_std,coef_x,coef_y


    def cal_flow_rec(self,flow_rec,moving_prob,p1,p2_reproj):
        A0_mean,A0_std,A1_mean,A1_std,bx_mean,bx_std,by_mean,by_std,coef_x,coef_y = self.linear_estimation(p1,p2_reproj)
        flow = flow_rec.copy()
        for i in range(flow_rec.shape[1]): # x 
            for j in range(flow_rec.shape[0]): # y
                if moving_prob[j,i] == 1:
                    flow[j,i,0] = coef_x.dot([(i-A0_mean)/A0_std,(j-A1_mean)/A1_std,1])*bx_std + bx_mean
                    flow[j,i,1] = coef_y.dot([(i-A0_mean)/A0_std,(j-A1_mean)/A1_std,1])*by_std + by_mean
        return flow

        
    def bundle_adjustment(self,R, T, K, _3d_pts, p1):

        def jac_sparse(_3d_pts):
            m = len(_3d_pts.T)*2
            n = len(_3d_pts.T)*3
            A = lil_matrix((m,n), dtype=int)
            for i in range(3):
                A[np.arange(m/2)*2, np.arange(len(_3d_pts.T))*3 + i] = 1
                A[np.arange(m/2)*2 + 1, np.arange(len(_3d_pts.T))*3 + i] = 1
            return A

        def error_fun(x,p1,K,R,T):
            _3d_pts_opt = x.reshape((len(p1), 3))
            p,J = cv2.projectPoints(_3d_pts_opt, R, T, K, np.array([]))
            p = p[:,0,:]
            error = (p1 - p).ravel()
            return error

        x0 = _3d_pts.T.ravel()
        A = jac_sparse(_3d_pts)
        res = least_squares(error_fun, x0, jac_sparsity=A, x_scale='jac', ftol=1e-8, method='trf',args=(p1,K,R,T))
        _3d_pts_opt = res.x.reshape((len(p1), 3))
        
        return _3d_pts_opt

    # def bundle_adjustment(self,R, T, K, _3d_pts, p1, p2):

    #     def jac_sparse(_3d_pts):
    #         m = len(_3d_pts.T)*2
    #         n = 6 + len(_3d_pts.T)*3
    #         A = lil_matrix((m,n), dtype=int)
    #         for i in range(3):
    #             A[np.arange(m/2)*2, 6 + np.arange(len(_3d_pts.T))*3 + i] = 1
    #             A[np.arange(m/2)*2 + 1, 6 + np.arange(len(_3d_pts.T))*3 + i] = 1
    #         return A

    #     def error_fun(x,p1,p2,K):
    #         R_init = np.eye(3, 3)
    #         T_init = np.zeros((3,1))
    #         R,T,_3d_pts_opt = x[:3], x[3:6], x[6:].reshape((len(p1), 3))
    #         p1_proj,J = cv2.projectPoints(_3d_pts_opt, R_init, T_init, K, np.array([]))
    #         p2_proj,J = cv2.projectPoints(_3d_pts_opt, R, T, K, np.array([]))
    #         p1_proj = p1_proj[:,0,:]
    #         p2_proj = p2_proj[:,0,:]
    #         error = abs(((p1 - p1_proj)+(p2 - p2_proj))).ravel()
    #         #print(error[:2])
    #         return error
       
    #     x0 = np.hstack((cv2.Rodrigues(R)[0].ravel(), T.ravel(),_3d_pts.T.ravel()))
    #     A = jac_sparse(_3d_pts)
    #     res = least_squares(error_fun, x0, jac_sparsity=A, x_scale='jac', ftol=1e-8, method='trf',args=(p1,p2,K))
    #     R,T,_3d_pts_opt = res.x[:3], res.x[3:6], res.x[6:].reshape((len(p1), 3))
    #     R = cv2.Rodrigues(R)[0]
    #     return R,T,_3d_pts_opt

    def cal_structure(self,flow_rec,K_cam, A, T_4x):
        _, _1t2 = R_t(T_4x[2])
        _1A2 = np.linalg.inv(A[1])
        g12 = epipolar_structure_from_flow(K_cam, flow_rec[:,:,:2], _1A2, _1t2)
        return g12

    def cal_structure_gt(self,flow,inter_depth,R_cam,T_cam,K_cam,s_ground,A,T_4x,n):
        scale = np.zeros((flow.shape[0],flow.shape[1],3))
        scale[:,:,0] = inter_depth
        scale[:,:,1] = inter_depth
        scale[:,:,2] = inter_depth
        x = unproject(pixelgrid(flow.shape[:2]), K_cam)
        _3d_pts_gt = x * scale
        _3d_pts_gt.reshape((_3d_pts_gt.shape[0]*_3d_pts_gt[1],3))
        s_gt = 1/n.dot(_3d_pts_gt.T)
        s_gt = np.percentile(s_gt, 5)/s_ground
        p2_gt,_ = cv2.projectPoints(_3d_pts_gt*s_gt, R_cam, T_cam, K_cam, np.array([]))
        p2_gt = p2_gt[:,0,:]
        p1_gt = np.zeros((p2_gt.shape[0],p2_gt.shape[1]))
        p1_gt[:,0] = pixelgrid(flow.shape[:2])[:,:,1]
        p1_gt[:,1] = pixelgrid(flow.shape[:2])[:,:,0]
        flow_gt = flow.copy()
        p1_gt = p1_gt.astype('int')
        flow_gt[p1_gt.T[1],p1_gt.T[0],:2] = p2_gt - p1_gt 
        g_gt = self.cal_structure(self,flow_gt,K_cam, A, T_4x)
        return g_gt
    
    def cal_F(self,R_cam,T_cam,K_cam):
        E_cam = skew(T_cam).dot(R_cam)
        F = np.linalg.inv(K_cam).T .dot(E_cam) .dot(np.linalg.inv(K_cam))
        return F

    def norm(self,a):
        return np.linalg.norm(a, axis=-1)

    def get_epipolar_error(self,p1, p2): ####degenrate case will fail
        F = self.cal_F(self.R_cam,self.T_cam,self.K_cam)
        p1 = to_homogenous(p1)
        p2 = to_homogenous(p2)
        l2 = p1 @ F.T
        l2_norm = self.norm(l2[:,:2])
        
        l1 = p2 @ F
        l1_norm = self.norm(l1[:,:2])
        
        #print((l2@p2.T).shape)
        
        d_epi = (abs(l2@p2.T).diagonal()/l2_norm+abs(l1@p1.T).diagonal()/l1_norm)/2
        
        return d_epi

    def filter_rigid_points(self,p1,p2):
        d_epi = self.get_epipolar_error(p1, p2)
        threshold = np.percentile(d_epi,10)
        mask = (d_epi > threshold).astype('uint8')
        p1 = p1[np.where(mask!=0)[0],:]
        p2 = p2[np.where(mask!=0)[0],:]
        print('Matches after filter: {}'.format(len(p1)))
        return p1,p2
    
    def interpolation_structure(self,g,search_size,p1,moving_prob):
        h,w = g.shape[:2]
        g_inter = np.zeros((h,w))
        g_inter[p1.T[1],p1.T[0]] = g[p1.T[1],p1.T[0]]
        for i in range(w):#x
            for j in range(h):#y
                if moving_prob[j,i] == 1 and g_inter[j,i] == 0:
                    if j-search_size > 0 and j+search_size < h and i-search_size > 0 and i+search_size < w:
                        search_area = g_inter[j-search_size:j+search_size,i-search_size:i+search_size]
                        num = np.count_nonzero(search_area)
                        g_inter[j,i] = search_area.sum()/num
        g[moving_prob==1] = g_inter[moving_prob==1]
        return g

    def get_reproj_error(self,_3d_pts,R_obj,T_obj,K,p2):
        p2_proj,J = cv2.projectPoints(_3d_pts, R_obj, T_obj, K, np.array([]))
        error = abs(p2 - p2_proj).sum()
        return error
