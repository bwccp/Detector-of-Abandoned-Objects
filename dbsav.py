''' 
Double background subtractor class DbsAverage for cpu only.
Version 09.10.2025. Latest changes 21.11.2025 
'''

import numpy as np
import cv2 as cv

class DbsAverage: 
    
    def __init__(self, histlen1, stride1, histlen2, stride2, h, w, c): 
        '''
        Class constructor for double background subtractor
        Parameters: 
        histlen1, histlen2 - number of frames in background 1 and 2;
        stride1, stride2 - steps between frames placed to hist1 and hist2;
        h, w, c - hight, width and channels in the frames. 
        '''
        
        self.histlen1 = histlen1  # influences the time of renewal of background 1
        self.histlen2 = histlen2  # influences the time of renewal of background 2
        self.stride1 = stride1    # influences the time of renewal of background 1
        self.stride2 = stride2    # influences the time of renewal of background 2
        self.fn = 0     # frames counter
        self.fih1 = 0   # frame index in self.hist1 
        self.fih2 = 0   # frame index in self.hist2 
        self.nfs1 = 0   # number of frames in self.sumfr1
        self.nfs2 = 0   # number of frames in self.sumfr2
        
        self.hist1 = np.zeros((histlen1, h, w, c), dtype=np.uint8)  # frames buffer 
        self.hist2 = np.zeros((histlen2, h, w, c), dtype=np.uint8)  # background 1 buffer
        self.sumfr1 = np.zeros((h, w, c), dtype=np.int32)           # sum of frames for foreground 1
        self.sumfr2 = np.zeros((h, w, c), dtype=np.int32)           # sum of background for foreground 2
        self.bg1 = np.zeros((h, w, c), dtype=np.uint8)              # background 1 
        self.bg2 = np.zeros((h, w, c), dtype=np.uint8)              # background 2 
        # Set the number of CPU cores to be used for computations 
        cv.setNumThreads(1)
        
    def apply(self, frame): 
        '''
        Produces foreground 2 that reflects changes in background 1.
        Should be called every video frame. 
        '''
        
        # Fast background modelling 
        self.fg1 = cv.absdiff(frame, self.bg1)  # can be omitted 
        if self.fn % self.stride1 == 0: 
            if self.nfs1 == 0:
                self.sumfr1 = frame.copy().astype(np.int32) 
                self.hist1[self.fih1, :, :, :] = frame.copy()
                self.bg1 = frame.copy()
                self.nfs1 += 1    
                self.fih1 += 1
            elif self.nfs1 > 0: 
                self.sumfr1 = self.sumfr1 + frame.astype(np.int32) - self.hist1[self.fih1, :, :, :] 
                self.hist1[self.fih1, :, :, :] = frame.copy()
                if self.nfs1 < self.histlen1: 
                    self.nfs1 += 1
                if self.fih1 < self.histlen1 - 1:
                    self.fih1 += 1
                elif self.fih1 == self.histlen1 - 1:
                    self.fih1 = 0
                self.bg1 = (self.sumfr1 / self.nfs1).astype(np.uint8)
        
        # Slower background modelling  
                
        self.fg2 = cv.absdiff(self.bg1, self.bg2)   
        if self.fn % self.stride2 == 0: 
            if self.nfs2 == 0:
                self.sumfr2 = self.bg1.copy().astype(np.int32) 
                self.hist2[self.fih2, :, :, :] = self.bg1.copy()
                self.bg2 = self.bg1.copy()
                self.nfs2 += 1    
                self.fih2 += 1
            elif self.nfs2 > 0: 
                self.sumfr2 = self.sumfr2 + self.bg1.astype(np.int32) - self.hist2[self.fih2, :, :, :] 
                self.hist2[self.fih2, :, :, :] = self.bg1.copy()
                if self.nfs2 < self.histlen2: 
                    self.nfs2 += 1
                if self.fih2 < self.histlen2 - 1:
                    self.fih2 += 1
                elif self.fih2 == self.histlen2 - 1:
                    self.fih2 = 0
                self.bg2 = (self.sumfr2 / self.nfs2).astype(np.uint8)
        
        self.fn += 1 
        return self.fg2  
          
          
          
