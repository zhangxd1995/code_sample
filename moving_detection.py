import cv2
import numpy as np
from util import *
from scipy import special 

class MovingDetection(object):

    def __init__(self,flow_x4,prediction,T_x4,A_,K_cam):
        #self.A = A
        self.flow = flow_x4
        self.prediction = prediction
        self.T_x4 = T_x4
        self.A_ = A_
        self.K_cam = K_cam        

    def on_batch(self):
        objs = self.isolate_objects(self.prediction)
        moving_probs = []
        prob = self.epi_plus_fvb(self.flow,self.T_x4,self.prediction,self.K_cam,fb=0)
        for obj in objs:
            prob_obj = obj * prob
            if prob_obj.sum()/obj.sum() >= 0.5:
                prob_obj[obj==1] = 1
                moving_probs.append(prob_obj)
            elif prob_obj.sum() > 500:
                 moving_probs.append(prob_obj)
        return moving_probs

    def isolate_objects(self,prediction): #prediction,moving_label):
        moving_labels_seg = [11,13,14,15,16]
        ## filter static predictions
        isolated_objects = []
        # prediction_move = prediction.copy()
        # prediction_move[prediction < 11] = 0
        ## use prediction as image to find contours
        for label in moving_labels_seg:
            prediction_move = prediction.copy()
            prediction_move[prediction != label] = 0
            if prediction_move.sum() > 500:
                img = prediction_move.copy()
                img = img.astype('uint8')
                ret, thresh = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((5, 5), np.uint8)
                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                erosion = cv2.erode(closing,kernel,iterations = 1)
                dilation = cv2.dilate(erosion,kernel,iterations = 1)
                contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # external_index = []
                # for i in range(len(contours)):
                #     if hierarchy[0][i][3] == -1:
                #         external_index.append(i)
                # contours_external = []
                # for index in external_index:
                #     contours_external.append(contours[index])
                
                ## isolated objects and filter small objects
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    mask = np.zeros(img.shape,np.uint8)
                    cnt = contours[i]
                    result = cv2.drawContours(mask,[cnt],0,1, -1)
                    if area > 500: ## if area too small, ignore it
                        isolated_objects.append(result)
        move_together = prediction.copy()
        move_together[prediction < 12] = 0
        move_together[prediction == 13] = 0
        move_together[prediction == 14] = 0
        move_together[prediction == 15] = 0
        move_together[prediction == 16] = 0
        #move_together = ((prediction== 12) or (prediction == 17) or (prediction == 18)).astype('int')
        if move_together.sum() > 500:
            img = move_together.copy()
            img = img.astype('uint8')
            ret, thresh = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            erosion = cv2.erode(closing,kernel,iterations = 1)
            dilation = cv2.dilate(erosion,kernel,iterations = 1)
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # external_index = []
            # for i in range(len(contours)):
            #     if hierarchy[0][i][3] == -1:
            #         external_index.append(i)
            # contours_external = []
            # for index in external_index:
            #     contours_external.append(contours[index])
            
            ## isolated objects and filter small objects
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                mask = np.zeros(img.shape,np.uint8)
                cnt = contours[i]
                result = cv2.drawContours(mask,[cnt],0,1, -1)
                if area > 500: ## if area too small, ignore it
                    isolated_objects.append(result)
                            
        return isolated_objects

    def cal_flow_residual(self,flow,H):
        p = pixelgrid(flow.shape[:2])
        p_r = register(p+flow[:,:,:2],H)
        flow_residual = p_r - p
        return flow_residual

    def cal_flow_trans(self,flow,T,K_cam):
        R_F = K_cam.dot(T[:3,:3]).dot(np.linalg.inv(K_cam))
        p = pixelgrid(flow.shape[:2])
        p_r = register(p,R_F)
        flow_trans = p+flow[:,:,:2] - p_r
        return flow_trans

    # def cal_motion_direction_prob(self, flow_residual, T_x4, K_cam, sigma=1.0,prior_rigid=0.5, prior_nonrigid=0.5):
    
    #     t = T_x4[2][:3,3]
    #     e = epipole(K_cam, t.T)
    #     p = pixelgrid(flow_residual.shape[:2])
               
    #     # Compute the distance of match and the angle between match and FoE.
    #     dist = norm(flow_residual)
        
    #     ang_foe = np.arctan2((e-p)[:,:,0],(e-p)[:,:,1])
    #     ang_uv = np.arctan2(flow_residual[:,:,0],flow_residual[:,:,1])
    #     ang = ang_uv - ang_foe
    
    #     # Compute probability that p' points towards epipole
    #     dist_from_line = dist * np.sin(ang)
        
    #     p = 1.0 / np.sqrt(2*np.pi*sigma**2) * np.exp( - dist_from_line**2 / (2*sigma**2))
        
    #     # Normalization constant
    #     # Note that we use special.ive = exp(-x) * special.iv(x) for numerical stability.
    #     c = np.sqrt(2 * np.pi) / sigma * special.ive(0,dist**2 / (sigma**2 * 4))
                
    #     # Probability that point is rigid is given as
    #     # p(rigid) = p(point|rigid) / ( p(point|rigid) + p(point|nonrigid)).
    #     prob = prior_rigid * p / (prior_rigid * p + prior_nonrigid * c / (2*np.pi))
    #     prob = 1 - prob
    #     return prob
    
    def cal_fvb_prob(self,T_x4,flow_x4,prediction,K_cam,forward):
        prob = np.zeros(flow_x4[0].shape[:2])
        if forward == 1:
            R_cam,T_cam = R_t(T_x4[0])
            t = T_x4[2][:3,3]
            e = epipole(K_cam, t.T)
            flow_trans = self.cal_flow_trans(flow_x4[0],T_x4[0],K_cam)
        else:
            R_cam,T_cam = R_t(T_x4[1])
            t = T_x4[3][:3,3]
            e = epipole(K_cam, t.T)
            flow_trans = self.cal_flow_trans(flow_x4[1],T_x4[1],K_cam)
        p = pixelgrid(flow_trans.shape[:2])
        d_min = norm(p - e) #+ norm(e/80)
        d_max = norm(p - e) + norm(e/0.1)
        d_mean = (d_max + d_min) / 2
        d_range = (d_max - d_min) / 2
        d = norm(p+flow_trans - e)
        prob = 1 - 1 / (1 + ((d - d_mean)/d_range)**20)
        prob[prediction<11] = 0
        return prob  

    # def cal_moving_prob(self,T_x4,flow,K_cam,A_,objs):
    #     moving_probs = []
    #     #flow_trans = self.cal_flow_trans(flow,T_x4,K_cam)    
    #     flow_residual = self.cal_flow_residual(flow,np.linalg.inv(A_[1]))
    #     prob_direction = self.cal_motion_direction_prob(flow_residual, T_x4, K_cam, sigma=1.0,prior_rigid=0.5, prior_nonrigid=0.5)
    #     prob_direction[prob_direction < 0.5] = 0
    #     prob_fvb = self.cal_fvb_prob(T_x4,flow,K_cam,A_)
    #     prob_fvb[prob_fvb < 0.6] = 0
    #     for obj in objs:
    #         prob = np.zeros(flow.shape[:2])
    #         prob[obj == 1] = prob_direction[obj==1] + prob_fvb[obj == 1]
    #         prob[prob >= 0.5] = 1
    #         if prob.sum() / obj.sum() > 0.7:
    #             prob[obj==1] = 1
    #             moving_probs.append(prob)
    #         # if prob.sum() / obj.sum() > 0.7:
    #         #     moving_probs.append(prob)            
    #     return moving_probs 

    # def cal_moving_prob_parallax(self,objs):
    #     moving_probs = []
    #     A = np.linalg.inv(self.A_[1])
    #     for obj in objs:
    #         prob = np.zeros(shape=(self.flow.shape[0],self.flow.shape[1]))
            
    #         p = pixelgrid(self.flow.shape[:2])
    #         p_w = register(p + self.flow[:,:,:2], A)
    #         p = p[obj == 1]
    #         p_w = p_w[obj == 1]
            
    #         parallax = p_w - p
            
    #         ## only 10% near ground point will be considered
    #         cut = int(parallax.shape[0] / 10 * 9)
    #         if cut > 1000:
    #             parallax = parallax[::-1][:1000]
    #         else:
    #             parallax = parallax[cut:]
    #         parallax_sorted = np.zeros((parallax.shape[0]))
    #         parallax_sorted = abs(parallax[:,0]) + abs(parallax[:,1])
    #         parallax_sorted = np.sort(parallax_sorted)
    #         cut = int(len(parallax_sorted)/10)
    #         if parallax_sorted[:cut].mean() > 2:
    #             prob[obj==1] = 1
    #         else:
    #             prob[obj==1] = 0
    #         #print(parallax[10:20])    
    #         ## delete small area
    #         if prob.sum() > 50:
    #             moving_probs.append(prob)
    #     return moving_probs

    def get_epipolar_rigid(self,T_x4,K_cam,flow_x4,prediction,forward):
        if forward == 1:
            R_cam,T_cam = R_t(T_x4[0])
            E_cam = skew(T_cam).dot(R_cam)
            F = np.linalg.inv(K_cam).T .dot(E_cam) .dot(np.linalg.inv(K_cam))
            flow = flow_x4[0]
        else:
            R_cam,T_cam = R_t(T_x4[1])
            E_cam = skew(T_cam).dot(R_cam)
            F = np.linalg.inv(K_cam).T .dot(E_cam) .dot(np.linalg.inv(K_cam))
            flow = flow_x4[1]

        p1 = pixelgrid(flow.shape[:2])
        p2 = p1 + flow[:,:,:2]
        p1 = to_homogenous(p1)
        p2 = to_homogenous(p2)

        l2 = p1 @ F.T
        #l2 = np.dot(F,p1)
        l2_norm = np.linalg.norm(l2[:,:,:2],axis=2)

        #l2 = np.array([l2[0]/l2[2],l2[1]/l2[2],1])
        l1 = p2 @ F
        l1_norm =  np.linalg.norm(l1[:,:,:2],axis=2)
        #l1 = np.array([l1[0]/l1[2],l1[1]/l1[2],1])
        #print(l2)
        d_epi = (abs(np.sum(l2*p2,axis=2))/l2_norm+abs(np.sum(l1*p1,axis=2))/l1_norm)/2
        #d_epi = (abs(l2.dot(p2))+abs(l1.dot(p1)))*10000
        #d_epi = p2.dot(F).dot(p1)
        prob = np.exp(-d_epi)
        prob[prediction < 11] = 1
        #d_epi[prediction < 11] = 0
        return 1 - prob

    def epi_plus_fvb(self,flow_x4,T_x4,prediction,K_cam,fb=0):
        if fb == 0:
            prob_fvb = self.cal_fvb_prob(T_x4,flow_x4,prediction,K_cam,1)
            prob_epi = self.get_epipolar_rigid(T_x4,K_cam,flow_x4,prediction,1)
        else:
            prob_fvb_fwd = self.cal_fvb_prob(T_x4,flow_x4,prediction,K_cam,1)
            prob_epi_fwd = self.get_epipolar_rigid(T_x4,K_cam,flow_x4,prediction,1)
            
            prob_fvb_bwd = self.cal_fvb_prob(T_x4,flow_x4,prediction,K_cam,0)
            prob_epi_bwd = self.get_epipolar_rigid(T_x4,K_cam,flow_x4,prediction,0)
            
            prob_epi = (prob_epi_fwd + prob_epi_bwd)/2
            prob_fvb = (prob_fvb_fwd + prob_fvb_bwd)/2
            
        prob_epi[prob_epi<0.5] = 0
        prob_epi[prob_epi>=0.5] = 1
        prob_fvb[prob_fvb<0.5] = 0
        prob_fvb[prob_fvb>=0.5] = 1
        prob = prob_epi + prob_fvb
        prob[prob>=1] = 1
        return prob

    

# class MovingDetectionPixelLevel(object):

