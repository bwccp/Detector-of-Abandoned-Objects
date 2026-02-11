'''
Detector of Abandoned Objects - version 0 (folder dao0), no tracking. 
Instructions: create instance of the class Dao. Call method apply() every frame
to detect abandoned objects. Call method validate() to remove false detections. 
Latest updates 21.11.2025 - 03.12.2025 
'''  

import cv2 as cv
import numpy as np
import dbsav
import sys

class Dao(dbsav.DbsAverage):
    '''
    Detector of Abandoned Objects. Inherits from double background 
    subtractor class DbsAverage. Processes frames with dtype=np.uint8
    Changed name 11.11.2025, 21.11.2025. Introduced changes for development 12.11.2025. 
    Mask dilation / closing and edges detection added 13.11.2025. Algorithm additions 14-28.11.2025. 
    '''
    
    def __init__(self, hl1, st1, hl2, st2, h, w, c=3, thr=35, athr=100, maxnobj=30, elsize=(13,13)): 
        '''
        Class constructor. Parameters:
        hl1, hl2 - history length for the first and second background subtractors
        st1, st2 - stride for the background subtractors
        h, w, c - frame height, width, number of channels
        thr - threshold for foreground2 (0...255)
        athr - minimum area of objects (px**2)
        maxnobj - maximum number of objects in detections (int)
        hl1*st1 determines time object gets detected 
        hl2*st2 determines time object gets undetected  
        '''
        super().__init__(hl1, st1, hl2, st2, h, w, c)
        self.width = w
        self.height = h 
        self.thr = thr 
        self.athr = athr 
        self.maxnobj = maxnobj
        self.kernel = cv.getStructuringElement(2, ksize=elsize) 
        # detections are placed/updated in *1 and *2 intermittently 
        self.nobj1 = 0 # int
        self.nobj2 = 0 
        self.masks1 = np.zeros(shape=(h, w), dtype=np.uint16)  
        self.masks2 = np.zeros(shape=(h, w), dtype=np.uint16)
        self.bboxes1 = np.zeros(shape=(maxnobj, 5), dtype=np.uint16)  # [x,y,w,h,area] 
        self.bboxes2 = np.zeros(shape=(maxnobj, 5), dtype=np.uint16)  #  
        self.centroids1 = np.zeros(shape=(maxnobj, 2), dtype=np.float64) 
        self.centroids2 = np.zeros(shape=(maxnobj, 2), dtype=np.float64)  
        self.pcf1 = []    # objects edges pictures of all objects in self.masks1
        self.pcf0 = []    # corresponding edges in the frame 
        self.pcf2 = []    # objects edges images of objects in self.masks2 
        self.maxndo1 = 0 # maximum quontity of objects in stage 1
        self.maxndo2 = 0 # same in stage 2
        
        # tracing attributes
        self.hl_trc = 100       # history length for the tracer
        self.sttrc = 1          # stride between frames for the tracer
        self.maxnobj_trc = 10   # max quantity of objects the tracer can recognize
        self.nprops = 12        # quantity of properties of objects in the tracer
        self.thr_roac = 0.01    # threshold relative object area change (dS/S), float 
        self.thr_iou = 0.2      # minimum iou for backtracing 
        self.thr_nfp = 5        # min number of frames object present to get od 
        self.thr_rmp = 0.87      # threshold of the ratio of matched points (nmp / tnp)
        self.thr_nfa = 100       # maximum object absence before removal from pcache and podcache
        # next object order number (int)  
        self.order = 0
        
        # the main storage of objects data in time 
        self.odcache = np.zeros(shape=(self.hl_trc, self.maxnobj_trc, self.nprops), dtype=np.int32)
        # odcache props: fn, nobj, od, x, y, w, h, area, pi, nfp, nfa, phi
        # fn - frame number, nobj - objects in this frame, od - order number, x, y, w, h, area, 
        # pi - previous frame this object index, nfp - quantity of frames object is present (0 if absent), 
        # nfa - quantity of frames object is absent, phi - photo index in pcache
        # index of the current record in odcache  
        self.odc_ind = 0 
        # properties of objects inlisted into the photos cache
        self.podcache = np.zeros(shape=(self.maxnobj_trc, 12), dtype=np.int32)
        # properties: fn, nobj, od, x, y, w, h, area, pi, nfp, nfa, fnld ; fn -frame number when placed into pcache
        # pi - object index in that frame; fnld - frame number object was last detected    
        # list of object edges images - 1D ndarrays dtype=np.uint8
        self.pcache = []
        
        
    def apply(self, frame): 
        '''
        Obtain detections of abandoned items. Must be called every frame.
        Parameters: frame - video frame, must have shape (h, w, c). 
        Results: placed into attributes *1 : self.nobj1, self.masks1, self.bboxes1, self.centroids1 
        Modified 11.11.2025. Added filter by area 12.11.25. Updated 21.11.2025, 02.12.2025  
        '''
        print(f'[Dao.apply]: start') 
        fg2 = super().apply(frame)
        fg2gs = cv.cvtColor(fg2, cv.COLOR_BGR2GRAY)
        ret, fg2mask = cv.threshold(fg2gs, self.thr, 255, cv.THRESH_BINARY)
        fg2closed = cv.morphologyEx(fg2mask, cv.MORPH_CLOSE, self.kernel) 
        #retval, labels, stats, centroids = cv.connectedComponentsWithStats(self.fg2maskd, 
        #    connectivity=8, ltype=cv.CV_16U)
        #print(self.fg2)
        retval, labels, stats, centroids = cv.connectedComponentsWithStatsWithAlgorithm(fg2closed, 
            connectivity=8, ltype=cv.CV_16U, ccltype=cv.CCL_GRANA)
        print('stats:')        
        # print(f'retval={type(retval)}') # retval - int 
        #print(labels) # labels - 2D numpy array uint16 [[ ][ ]] each pixel value = label
        print(stats) # stats - 2D numpy array int32 [[x,y,w,h,area]]
        #print(centroids) # centroids - 2D numpy array float64 [x, y]
        # contours, hierarchy = cv.findContours(fg2mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # nobj = len(contours)
        # List of bboxes 
        # bboxes = [] 
        # for i in range(nobj): 
        #    if cv.contourArea(contours[i]) > self.athr: 
        #        x,y,w,h = cv.boundingRect(contours[i])
        #        bbox = [x, y, x + w, y + h]
        #        bboxes.append(bbox.copy())
                 
        # Removing objects with area below the threshold and background - results in attributes self.*1 
        self.filterByArea(retval, labels, stats, centroids, self.athr) 
        if self.nobj1 > self.maxndo1: 
            self.maxndo1 = self.nobj1
             
        # extracting data for the pure bboxes
        #bboxes = []                    
        #for bbox in self.bboxes1: 
        #    x1 = bbox[0]; y1 = bbox[1]; w = bbox[2]; h = bbox[3]
        #    bboxes.append([x1, y1, w, h])
        # return bboxes
        print(f'[.apply]: end') 
        
    def evaluate(self, img):   # definition changed 02/12/2025
        '''
        Checks detections. Remove detections if object(s) are not present. 
        Created 11.11.2025. Detect edges at the object boundaries. 
        img - BGR ndarray dtype=np.uint8, 
        Params: img - image of frame, 3-channel 8-bit image
        Return: void - nothing
        Results are in attributes self.nobj2, self.masks2, self.bboxes2, self.centroids2 
        13.11.2025 - bug corrections, visualization, expanded bboxes by 5 px.
        14.11.2025 refactoring, modifications; 17.11.2025 major revision 21,28.11.2025 updates 
        03.12.2025 debugging
        '''
        print(f'[Dao.evaluate]: start') 
        # clear output numpy arrays 
        self.masks2.fill(0) 
        self.bboxes2.fill(0) 
        self.centroids2.fill(0) 
        # images initialization for demonstrations 
        #crop0 = np.zeros(shape=(50, 50))
        #crop1 = np.zeros(shape=(50, 50))
        #edges0 = np.zeros(shape=(50, 50))
        #edges1 = np.zeros(shape=(50, 50))
        #print('self.bboxes1:', self.bboxes1)
        # objects counter 
        objc = 0
        # empty lists of object edges pictures
        self.pcf0.clear()
        self.pcf1.clear()
        print(f'Input self.nobj1={self.nobj1}')
        for i in range(self.nobj1): 
            x1 = self.bboxes1[i, 0]
            if x1 < 5: 
                x1 = 0 
            else: 
                x1 = x1 - 5
            y1 = self.bboxes1[i, 1] 
            if y1 < 5: 
                y1 = 0 
            else: 
                y1 = y1 - 5    
            x2 = x1 + self.bboxes1[i, 2] 
            if x2 > self.width - 11: 
                x2 = self.width - 1
            else:
                x2 = x2 + 10    
            y2 = y1 + self.bboxes1[i, 3] 
            if y2 > self.height - 11:
                y2 = self.height - 1
            else:
                y2 = y2 + 10    
            print(f'bbox for the crop: x1={x1} y1={y1} x2={x2} y2={y2}')    
            crop0 = img[y1:y2, x1:x2]           # crop of the frame
            crop1 = self.masks1[y1:y2, x1:x2]   # crop of the masks
            
            crop1[crop1 == i] = 255
            crop1 = crop1.astype(np.uint8)
            edges0 = cv.Canny(crop0, 100, 200)  # edges in the frame
            edges1 = cv.Canny(crop1, 100, 200)  # edges of the mask
            self.pcf0.append(edges0.copy()) 
            self.pcf1.append(edges1.copy()) 
            #rep = cv.matchShapes(edges0, edges1, cv.CONTOURS_MATCH_I1, 0) # not well for this task
            # number of matched points, total number of points in the stencil image
            nmp, tnp = self.matchEdge(edges1, edges0)
            # ratio of matched points 
            rmp = nmp / (tnp + 0.001) 
            print(f'Object {i} nmp={nmp}, tnp={tnp}, rmp={rmp}, thr_rmp={self.thr_rmp}') 
            if rmp >= self.thr_rmp:
                objc += 1
                self.masks2[self.masks1 == i] = objc
                self.bboxes2[objc - 1, :] = self.bboxes1[i, :]
                self.centroids2[objc - 1, :] = self.centroids1[i, :]
                self.pcf2.append(edges1)  # edges pictures dtype=np.uint8 
        self.nobj2 = objc
        if self.nobj2 > self.maxndo2:
            self.maxndo2 = self.nobj2
        print(f'obj in pcf2: {len(self.pcf2)}') 
        print(f'Output self.nobj2={self.nobj2}')
        print(f'[Dao.evaluate]: end') 
        # return images of a single object for development purposes     
        #return crop0, crop1, edges0, edges1 
        
    def trace(self, frame): 
        '''
        DO NOT USE! NONFUNCTIONAL AND CAUSES DAMAGE 
        Perform tracing of stationary objects in time. 
        Should be called every frame of video 
        Created 11.11.2025. Started dev 17.11.2025. Expanded 18,19,21.11.2025
        Revised 26-28.11.2025, 1.12.2025 
        Parameters: frame - current video frame image 
        Output: tracing results for the current frame objects - ndarray 
        [id, x, y, w, h]
        '''
        
        print(f'[.trace]: start')
        # clear the current frame data 
        self.odcache[self.odc_ind, :, :] = 0 
        # copy current frame true detections to odcache 
        self.odcache[self.odc_ind, 0:self.nobj2, 0] = self.fn                # frame number
        self.odcache[self.odc_ind, 0:self.nobj2, 1] = -1                     # object id before assignment 
        self.odcache[self.odc_ind, 0:self.nobj2, 3:8] = self.bboxes2[0:self.nobj2, :]   # [x,y,w,h,area] for detected objects 
        self.odcache[self.odc_ind, 0:self.nobj2, 8] = -1                     # pi - previous frame object index
        self.odcache[self.odc_ind, 0:self.nobj2, 11] = -1                    # phi - object index in pcache and podcache 
        # obtain previous frame data index 
        if self.odc_ind == 0:
            iopf = self.maxnobj_trc - 1
        else:
            iopf = self.odc_ind - 1    
        # tracing objects from the previous frame to the current frame according to min iou 
        # quantity of objects in the previous frame data 
        qopf = self.odcache[iopf, 0, 1] 
        # quantity of objects in the current frame data
        qocf  = self.odcache[self.odc_ind, 0, 1]
        # iou matrix definition -obsolete 
        #ioum = np.zeros[shape=(qocf, qopf), dtype=np.float32] 
        # cycle through objects in the current frame for their tracing  
        for i in range(qocf): 
            # carrying out backtracing 
            bboxc = self.odcache[self.odc_ind, i, 3:7] 
            miniou = 0
            objind = -1
            for j in range(qopf):
                bboxp = self.odcache[iopf, j, 3:7]        
                iou = self.getiou(bboxc, bboxp)
                if iou < self.thr_iou:
                    continue 
                if j == 0:
                    miniou = iou
                    objind = 0
                if j > 0:
                    if iou < miniou:   # iou minimum value 
                        miniou = iou  
                        objind = j    # object index in the previous frame 
            # treatment of backtracing results             
            # determine if object is included in pcache/podcache - can be moved to just below the for i 
            pcacheind = self.search_pcache(i, 0.9) 
            if pcacheind > -1:
                print(f'Object {i} found in pcache. Adding it to the tracking data.')
                # object i was found in pcache/podcache - assign all tracking data  
                pcod = self.podcache[pcacheind, 2]  ## order number
                self.odcache[self.odc_ind, i, 2] = pcod  # assign od
                pi = self.podcache[pcacheind, 8]   # previous frame object index
                self.odcache[self.odc_ind, i, 8] = pi  # assign pi
                nfp = self.odcache[iopf, pi, 8] + 1  # 
                self.odcache[self.odc_ind, i, 9] = nfp  # 
                self.podcache[pcacheind, 9] = nfp  # frames object is present
                self.odcache[self.odc_ind, i, 10] = 0  # zero frames object is absent 
                self.podcache[pcacheind, 10] = 0   # zero frames object is absent
                self.odcache[self.odc_ind, i, 11] = pcacheind  # pcache index 
                self.podcache[pcacheind, 11] = self.fn   # last frame object was detected 
                # proceed to the next object in the current frame - can be moved to just below the for i 
                continue 
                        
            if pcacheind == -1: 
                # object i was not found in pcache/podcache using its edges picture   
                if objind > -1: 
                    print(f'object {i} not in pcache, found in the previous frame via location data')
                    self.odcache[self.odc_ind, i, 8] =  objind   # assigning pi   
                    nfp = self.odcache[iopf, objind, 9] + 1      # frames object is present  
                    self.odcache[self.odc_ind, i, 9] = nfp       # assign nfp for object in the current frame 
                    self.odcache[self.odc_ind, i, 10] = 0        # nfa = 0 not absent 
                    pfid = self.odcache[iopf, objind, 2]         # prev frame id of object i in the previous frame
                    # case object already has been assigned id 
                    if pfid > -1: 
                        self.odcache[self.odc_ind, i, 2] = pfid  #  id for object i in the current frame
                        # search of object with id=pfid in podcache/pcache using id 
                        nobjpc = self.podcache[0, 1]  # number of objects in podcache/pcache
                        pcind = -1  # index of object in pcache/podcache 
                        for j in range(nobjpc): 
                            id = self.podcache[j, 2]
                            if id == pfid: 
                                # found object in pcache
                                pcind = j
                                break
                        if pcind > -1:        
                            print(f'[.trace]: Object {i} was found in podcache[] via id={pfid} index {pcind} ')         
                            print('it was not found through search_pcache()')
                            print(f'Check the correctness of assigning object in frame to object in pcache') 
                            #  data for object in odcache , podcache 
                            self.odcache[self.odc_ind, i, 11] = pcind
                            self.podcache[pcind, 11] = self.fn    # updating fnld 
                            self.podcache[pcind, 9] = nfp
                            self.podcache[pcind, 10] = 0   # nfa
                            continue
                        
                        if pcind == -1:
                            print(f'object {i} having id was not found in podcache, pcache - not possible scenario') 
                            # only objects having od are placed into pcache / podcache 
                            # check criteria for giving od and placing in pcache, podcache
                            # area change, nfp
                            # get area change of the object
                            areap = self.odcache[iopf, objind, 7]
                            areac = self.odcache[self.odc_ind, i, 7]
                            roac = np.fabs((areac - areap) / areac)      # relative object area change, float
                            if roac < self.thr_roac and nfp >= self.thr_nfp:
                                # ready to give od 
                                objod = self.order
                                self.order += 1
                                self.odcache[self.odc_ind, i, 2] = objod 
                                ## and place object edges picture into cache
                                self.pcache.append(self.pcf[i].copy())
                                # update number of objects in podcache and pcache 
                                nobjpc = self.podcache[0, 1] + 1    # current quantity of objects in pcache
                                if nobjpc == self.maxnobj_trc: 
                                    print(f'[.trace]: Error - quantity of objects in podcache exceeds the set value {self.maxnobj_trc}. Increase and run again.') 
                                    sys.exit()
                                self.podcache[0:nobjpc, 1] = nobjpc  # update number of objects in podcache 
                                self.podcache[nobjpc - 1, 0] = self.fn  # frame number the edges picture added to pcache
                                self.podcache[nobjpc - 1, 2] = objod    # od
                                self.podcache[nobjpc - 1, 3:8] = self.odcache[self.odc_ind, i, 3:8] # copy x,y,w,h,area 
                                self.podcache[nobjpc - 1, 8] = i    # object index in    
                                self.podcache[nobjpc - 1, 9] = nfp   # n frames object present
                                self.podcache[nobjpc - 1, 10] = 0     # n frames absent
                                self.podcache[nobjpc - 1, 11] = self.fn  # fn last detected
                            # if the criteria not satisfied - proceed to the ne4xt frame     
                            continue
                        
                    if pfid == -1: 
                        print(f' object {i} was not given od yet, found in previous frame')  
                        # check criteria for giving od and placing in pcache, podcache
                        # area change, nfp
                        # get area change of the object
                        areap = self.odcache[iopf, objind, 7]
                        areac = self.odcache[self.odc_ind, i, 7]
                        roac = np.fabs((areac - areap) / areac)      # relative object area change, float
                        print(f'Relative object {i} area change {roac}') 
                        if roac < self.thr_roac and nfp >= self.thr_nfp:
                            print(f'Object {i} ready to be given od: nfp={nfp}, thr_roac={self.thr_roac}')
                            objod = self.order
                            self.order += 1
                            self.odcache[self.odc_ind, i, 2] = objod 
                            print(f'object {i} : place object edges picture into pcache')
                            self.pcache.append(self.pcf[i].copy())
                            # update number of objects in podcache and pcache 
                            nobjpc = self.podcache[0, 1] + 1    # current quantity of objects in pcache
                            if nobjpc == self.maxnobj_trc: 
                                print(f'[.trace]: Error - quantity of objects in podcache exceeds the set value {self.maxnobj_trc}. Increase and run again.') 
                                sys.exit()
                            self.podcache[0:nobjpc, 1] = nobjpc  # update number of objects in podcache 
                            self.podcache[nobjpc - 1, 0] = self.fn  # frame number the edges picture added to pcache
                            self.podcache[nobjpc - 1, 2] = objod    # od
                            self.podcache[nobjpc - 1, 3:8] = self.odcache[self.odc_ind, i, 3:8] # copy x,y,w,h,area 
                            self.podcache[nobjpc - 1, 8] = i    # object index in    
                            self.podcache[nobjpc - 1, 9] = nfp   # n frames object present
                            self.podcache[nobjpc - 1, 10] = 0     # n frames absent
                            self.podcache[nobjpc - 1, 11] = self.fn  # fnld 
                        # if the criteria not satisfied - proceed to the ne4xt frame     
                        continue
                                                
                if objind == -1:
                    print(f'object {i} was not found in the previous frame as to bbox, nfp set to 1') 
                    self.odcache[self.odc_ind, i, 9] = 1  # nfp - present for 1 frame
                    continue        
                
                
                #################                                        
            '''        if pcacheind == -1:
                        # object i was not found in pcache - assign new od, place to pcache     
                        self.odcache[self.odc_ind, i, 2] = self.order  # od
                        self.order += 1      # set next od
                        # add a copy of object edges picture to pcache 
                        self.pcache.append(self.pcf[i].copy()) 
                        # update number of objects in podcache and pcache 
                        nobjpc = self.podcache[0, 1] + 1    # current quantity of objects in pcache
                        if nobjpc == self.maxnobj_trc: 
                            print(f'[.trace]: Error - quantity of objects in podcache exceeds the set value {self.maxnobj_trc}. Increase and run again.') 
                            sys.exit()
                        self.podcache[0:nobjpc, 1] = nobjpc
                        # frame number the edges added to pcache 
                        self.podcache[nobjpc - 1, 0] = self.fn
                        self.podcache[nobjpc - 1, 3:8] = self.odcache[self.odc_ind, i, 3:8] # copy x,y,w,h,area    
                        self.podcache[nobjpc - 1, 8] = 1   # n frames object present
                          
                    if od > -1:
                        # od already assigned - copy od and some data of object i: phi - should run even if area unstable
                        self.odcache[self.odc_ind, i, 2] = od
                        pcoi = self.odcache[iopf, objind, 11]  # phi - podcache and pcashe index exists for objects with od
                        self.odcache[self.odc_ind, i, 11] = pcoi
                        self.podcache[pcoi, 8] += 1  # nfp 
                        self.podcache[pcoi, 9] = 0   # nfa      
            if objind == -1:    
                # object was not present in the previous frame - wait for more frames to compute dS/S
                self.odcache[self.odc_ind, i, 9] = 1
                # search object in pcache 
            '''    
        # end of for i - cycle through detected objects in this frame        
        # scan through pcache / podcache  
        nobjpc = self.podcache[0, 1]  # number of objects in podcache/pcache
        # cycle over objects in pcache, podcache
        i = 0 
        while i < nobjpc: 
            if self.podcache[i, 11] == self.fn: 
                # object was detected in this frame (fnld==self.fn)
                i += 1
                continue
            # check object presence in this frame using edges picture - take object picture self.pcache[i]
            # take bbox crop of the frame edges, compute matching score of object 
            x1 = self.podcache[i, 3] - 5 
            y1 = self.podcache[i, 4] - 5 
            x2 = x1 + self.podcache[i, 5] + 10 
            y2 = y1 + self.podcache[i, 6] + 10 
            crop = frame[y1:y2, x1:x2]           # crop of the frame
            edges = cv.Canny(crop, 100, 200)
            # number of matched points, total number of points in the stencil image
            nmp, tnp = self.matchEdge(self.pcache[i], edges)
            # ratio of matched points 
            rmp = nmp / (tnp + 0.01)
            print(f'object {i} in pcache has rmp={rmp} with this frame edges in crop')
            if rmp >= self.thr_rmp: 
                print(f'object pcache{i} is detected via edges! place detection into the odcache[]')  
                nobj = self.odcache[self.odc_ind, 0, 1] + 1   # updated number of objects 
                self.odcache[self.odc_ind, 0:nobj, 1] = nobj  # 
                self.odcache[self.odc_ind, nobj - 1, 0] = self.fn 
                self.odcache[self.odc_ind, nobj - 1, 2:9] = self.podcache[i, 2:9]
                nfp = self.podcache[i, 9] + 1
                self.podcache[i, 9] = nfp  
                self.podcache[i, 10] = 0   
                self.odcache[self.odc_ind, nobj - 1, 9:10] = self.podcache[i, 9:11] 
                self.podcache[i, 11] = self.fn  
                i += 1
            else:
                # object i not detected - set object properties     
                self.podcache[i, 9] = 0    # number of frames present set to 0
                nfa = self.podcache[i, 10] + 1
                self.podcache[i, 10] = nfa  # number of frames object absent
                # delete object if absent too much 
                if nfa > self.thr_nfa: 
                    # delete from pcache
                    del self.pcache[i] 
                    # delete object i from podcache
                    for j in range(i, nobjpc - 2): 
                        self.podcache[j, :] = self.podcache[j+1, :]
                    self.podcache[nobjpc - 1, :] = 0 
                    # set new number of objects in pcache, podcache
                    nobjpc -= 1 
                    self.podcache[0:nobjpc, 1] = nobjpc
                    continue
                else: 
                    i += 1  
                      
        nobj = self.odcache[self.odc_ind, 0, 1]            
        # the next index of data in the odcache
        if self.odc_ind == self.hl_trc - 1:
            self.odc_ind = 0
        else:    
            self.odc_ind += 1
        print(f'[Dao.trace]: end') 
        return self.odcache[self.odc_ind, 0:nobj, 2:7] 
    
    def filterByArea(self, nobj, masks, bboxes, centroids, thr_area=100):
        '''
        filter by area
        1) Remove data of objects posessing area below the threshold
        2) Remove data of object corresponding to the whole frame
        Uses arguments produced by the method cv.connectedComponentsWithStats
        nobj (int), masks (ndarray uint16), bboxes(ndarray int32),
        centroids(float64), thr_area(int)
        
        Results are in attributes self.nobj1, self.masks1, self.bboxes1, 
        self.centroids1 - objects above the threshold, background not included 
        13.11.2025, refactoring 14.11.2025; algorithm change 17.11.2025 update 21.11.2025
        '''
        print(f'[Dao.filterByArea]: start') 
        # clear all results numpy arrays
        self.masks1.fill(0)
        self.bboxes1.fill(0)
        self.centroids1.fill(0)
        
        # counting objects with area > thr_area 
        n = 0 
        for bbox in bboxes:
            area = bbox[4]
            if area > self.athr: 
                n += 1 
        if n > self.maxnobj:
            print(f'Object with area above the threshold {n} exceeds maxnobj={self.maxnobj}')
            sys.exit() 
                    
        # objects counter
        objc = 0
        # filter out objects below the threshold and equal to the whole frame 
        for i in range(nobj): 
            area = bboxes[i, 4]
            x = bboxes[i, 0]
            y = bboxes[i, 1]
            w = bboxes[i, 2]
            h = bboxes[i, 3]
            #print(x,y,w,h)
            if x == 0 and y == 0 and w == self.width and h == self.height:
                continue
            if area > self.athr: 
                objc += 1
                if objc <= self.maxnobj: 
                    self.masks1[masks==i] = objc
                    self.bboxes1[objc - 1, :] = bboxes[i, :]
                    self.centroids1[objc - 1, :] = centroids[i, :]
                else:
                    print(f'quontity of object with area above the threshold exceeds the maxnobj')
                    sys.exit()    
                
        #print(f'self.bboxes1=\n{self.bboxes1}')        
        self.nobj1 = objc 
        print(f'self.nobj1={self.nobj1}')
        print(f'[Dao.filterByArea]: end') 
        
        
    def matchEdge(self, stencil, sample):
        '''
        How many pixels in sample coincide with 255 values pixels in stencil 
        stencil - uint8 1D ndarray 
        sample - same size and dtype
        Return: nmp, tnp ; nmp - number of matching points; tnp - total number
        of points with value 255 in the stencil  
        Created 13.11.2025, updated 21.11.2025
        '''
        h = stencil.shape[0] 
        w = stencil.shape[1] 
        nmp = 0  # number of matching pixels 
        tnp = 0  # total number of 255 pixels in stencil 
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if stencil[i, j] == 255:
                    tnp += 1
                    if sample[i, j] == 255:
                        nmp += 1
                    elif sample[i-1, j] == 255:
                        nmp += 1
                    elif sample[i+1, j] == 255:
                        nmp += 1
        return nmp, tnp
        
    def getiou(self, bbox1, bbox2):
        '''
        Computes intersection over union for two bboxes. 
        bbox1, bbox2 - numpy arrays of shape=(4,) containing x,y,w,h 
        Returns iou = S(intersection) / (S1 + S2 - S(intersection)) 
        Created and developed 18.11.2025
        '''
        x11 = bbox1[0]
        y11 = bbox1[1]
        x12 = x11 + bbox1[2]
        y12 = y11 + bbox1[3]
        
        x21 = bbox2[0] 
        y21 = bbox2[1] 
        x22 = x21 + bbox2[2] 
        y22 = y21 + bbox2[3] 
        
        xl = max(x11, x21) 
        xr = min(x12, x22) 
        yt = max(y11, y21) 
        yb = min(y12, y22) 
        
        dx = xr - xl
        dy = yb - yt
        if dx < 0 or dy < 0:
            Si = 0
        else: 
            Si = dx * dy 
        S1 = (x12 - x11) * (y12 - y11)
        S2 = (x22 - x21) * (y22 - y21)
        iou = Si / (S1 + S2 - Si)
        #print(Si, S1, S2)
        return iou 
        
    def search_pcache(self, objind1, thr=0.9): 
        '''
        Looks through the pcache pictures to find most suitable object.
        Params: objind1 - index of searched object in the detection results *2 (0,1,..)
        thr - threshold for matching
        Return: objind2 - index of object in pcache or -1 if not present in pcache
        
        '''
        
        objind2 = -1
        # x1 = self.bboxes2[objind1, 0] - 5 
        # y1 = self.bboxes2[objind1, 1] - 5 
        # x2 = x1 + self.bboxes2[objind1, 2] + 10 
        # y2 = y1 + self.bboxes2[objind1, 3] + 10 
        # crop0 = self.masks2[y1:y2, x1:x2]
        # crop0[crop0 == objind1 + 1] = 255
        # crop0 = crop0.astype(np.uint8)
        # edges0 = cv.Canny(crop0, 100, 200)
        edges0 = self.pcf[objind1]
        qopcache = self.podcache[0, 1]  # quantity of objects in pcache
        bestmatch = 0
        for i in range(qopcache): 
            nmp, tnp = self.matchEdge(edges0, pcache[i])
            rmp = nmp / (tnp + 0.01) 
            if rmp > bestmatch:
                bestmatch = rmp
                objind2 = i
        
        return objind2 
        
