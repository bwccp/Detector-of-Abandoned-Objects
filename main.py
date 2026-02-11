import cv2 as cv
import numpy as np
import middbsav, sys, time 

# input video file or stream 
#vfile = 'cor1obj1.mp4' 
vfile = 'outside1.mp4'
# whether to record output to video file 
record = True 
# output video file 
ovfile = 'outres1.mp4'
# scale for frame resizing
scale = 0.5

def main():
    '''
    Driver code for class middbsav.Dao() 
    Detection of objects that became nonmoving (detection of abandoned objects). 
    Modified 11.11.2025: No time measurements
    Modified 13.11.2025. Updated for removing false positives 14.11.2025. 
    Revised 28/11/2025. Changes 02.12.2025  
    '''
        
    vco = cv.VideoCapture(vfile)
    if not vco.isOpened():
        print('The video cannot be opened. Quitting.')
        sys.exit()
    # Obtain width, height of video frames    
    width = vco.get(cv.CAP_PROP_FRAME_WIDTH)
    height = vco.get(cv.CAP_PROP_FRAME_HEIGHT)
    vfps = vco.get(cv.CAP_PROP_FPS)
    # frame of half size is used to save memory and time
    w = int(width * scale)
    h = int(height * scale)
    ## Create instance of class MidDbsav
    detector = middbsav.Dao(hl1=110, st1=8, hl2=110, st2=8, h=h, w=w, c=3, thr=30, athr=9) 
    
    if record:    
        fourcc = cv.VideoWriter.fourcc(*'XVID') 
        vwo = cv.VideoWriter(ovfile, fourcc, vfps, (w, h))
    
    polyg_points=np.array([[770,431], [563, 308], [24, 311], [275, 534], [454, 533]], np.int32)
    polyg_points = polyg_points.reshape((-1,1,2))
    mask = np.zeros((h,w), np.uint8) 
    cv.fillPoly(mask, [polyg_points], 255)
    
    while True:
        print(f'Next frame processing start. maxndo1={detector.maxndo1}, maxndo2={detector.maxndo2}')
        ret, frame = vco.read()
        if ret == False:
            print('The next video frame cannot be obtained. Quitting.') 
            sys.exit()
        framer0 = cv.resize(frame, None, fx=scale, fy=scale)
        framer = cv.bitwise_and(framer0, framer0, mask=mask) 
        #t0 = time.perf_counter()
        ## call method apply every frame; the method returns numpy array with bboxes
        detector.apply(framer)
        # call method validate to remove false detections (false positives) 
        detector.evaluate(framer)
        
        cv.namedWindow('crop0') 
        cv.moveWindow('crop0', 60, 680)
        cv.namedWindow('crop1')
        cv.moveWindow('crop1', 460, 680)
        cv.namedWindow('crop2') 
        cv.moveWindow('crop2', 860, 680)
        cv.namedWindow('crop3')
        cv.moveWindow('crop3', 1260, 680)
        
        cv.namedWindow('edges0') 
        cv.moveWindow('edges0', 60, 880)
        cv.namedWindow('edges1')
        cv.moveWindow('edges1', 460, 880)
        cv.namedWindow('edges2') 
        cv.moveWindow('edges2', 860, 880)
        cv.namedWindow('edges3')
        cv.moveWindow('edges3', 1260, 880)
        
        cv.namedWindow('edges0m') 
        cv.moveWindow('edges0m', 60, 1080)
        cv.namedWindow('edges1m')
        cv.moveWindow('edges1m', 460, 1080)
        cv.namedWindow('edges2m') 
        cv.moveWindow('edges2m', 860, 1080)
        cv.namedWindow('edges3m')
        cv.moveWindow('edges3m', 1260, 1080)
                
        # preparing pictures according to nobj1 bboxes1
        nobjt = 0
        if detector.nobj1 >= 1:
            x1 = detector.bboxes1[0, 0] - 5
            y1 = detector.bboxes1[0, 1] 
            if y1 > 6:
                y1 = y1 - 5
            else:
                y1 = 0    
            x2 = x1 + detector.bboxes1[0, 2] + 10 
            y2 = y1 + detector.bboxes1[0, 3] + 10
            crop0 = framer[y1:y2, x1:x2]
            edges0 = detector.pcf0[0]
            edges0m = detector.pcf1[0]
            cv.imshow('crop0', crop0)
            cv.imshow('edges0', edges0)
            cv.imshow('edges0m', edges0m)
            if detector.nobj1 >= 2:
                x1 = detector.bboxes1[1, 0] - 5 
                y1 = detector.bboxes1[1, 1] 
                if y1 > 6:
                    y1 = y1 - 5
                else:
                    y1 = 0    
                x2 = x1 + detector.bboxes1[1, 2] + 10
                y2 = y1 + detector.bboxes1[1, 3] + 10
                crop1 = framer[y1:y2, x1:x2]
                edges1 = detector.pcf0[1] 
                edges1m = detector.pcf1[1] 
                cv.imshow('crop1', crop1)
                cv.imshow('edges1', edges1)
                cv.imshow('edges1m', edges1m) 
                if detector.nobj1 >= 3:
                    x1 = detector.bboxes1[2, 0] - 5
                    y1 = detector.bboxes1[2, 1] 
                    x2 = x1 + detector.bboxes1[2, 2] + 10 
                    y2 = y1 + detector.bboxes1[2, 3] + 10
                    crop2 = framer[y1:y2, x1:x2]
                    edges2 = detector.pcf0[2] 
                    edges2m = detector.pcf1[2] 
                    cv.imshow('crop2', crop2)
                    cv.imshow('edges2', edges2)
                    cv.imshow('edges2m', edges2m)
        #cv.imshow('crop1', crop1) 
        #cv.imshow('edges0', edges0) 
        #cv.imshow('edges1', edges1) 
        #dt1 = 1000 * (time.perf_counter() - t0)
        #print(f't={dt1:.3f}ms')
        print(detector.bboxes1[0:4, :])              
        print(f'Frame consecutive number: {detector.fn}, Quantity of detected objects: {detector.nobj2}') 
        #print(objdata)
         
        #plotting detections in the frame 
        for i in range(detector.nobj2): 
            x1 = detector.bboxes2[i, 0]
            y1 = detector.bboxes2[i, 1]
            x2 = x1 + detector.bboxes2[i, 2]
            y2 = y1 + detector.bboxes2[i, 3]
            cv.rectangle(framer0, (x1, y1), (x2, y2), (255,0,0), 1)
        #        print(bboxes[i, :]) 
        #        nobj += 1
        #print(f'Objects with area above the threshold {nobj}')
        #for bbox in bboxes:
        #    x1 = bbox[0]
        #    y1 = bbox[1]
        #    x2 = x1 + bbox[2]
        #    y2 = y1 + bbox[3]
        #    cv.rectangle(framer, (x1,y1), (x2,y2), (255,0,0), 1)
                    
        # show frame with bboxes
        cv.namedWindow(f'Video (scale={scale})') 
        cv.moveWindow(f'Video (scale={scale})', 70, 40)
        cv.namedWindow('Foreground mask, closed')
        cv.moveWindow('Foreground mask, closed', 1040, 40)  
        cv.imshow(f'Video (scale={scale})', framer0)
        cv.imshow('Foreground mask, closed', detector.fg2maskd)
        # cv.imshow('Background1', detector.bg1) # intermediate results for development
        # cv.imshow('Background2', detector.bg2) #   
        # cv.imshow('Foreground2', detector.fg2) #
        if record:
            vwo.write(framer0) 
        key = cv.waitKey(10)
        if key == ord('q'):
            print('Quitting after key "q" has been pressed')
            break
        if key == ord('p'):
            print('Pausing. Press "p" again to resume.')
            while True:
                key = cv.waitKey(50)
                if key == ord('p'):
                    print('Resuming.')
                    break    
    cv.destroyAllWindows()        
    vco.release()
    if record:
        vwo.release() 
    
if __name__ == '__main__':
    main() 
