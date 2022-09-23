import cv2
import depthai as dai
import numpy as np



def get_frame(queue):
    frame = queue.get()
    return frame.getCvFrame()

def get_mono_camera(pipline, isLeft):
    mono = pipline.createMonoCamera()

    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    return mono

def get_stereo_pair(pipeline, mono_left, mono_right):

    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    return stereo

def mouse_callback(event,x,y,flags,param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y

if __name__ == '__main__':
    mouse_x = 0
    mouse_y = 640

    pipeline = dai.Pipeline()
    mono_left = get_mono_camera(pipeline, isLeft=True)
    mono_right = get_mono_camera(pipeline, isLeft=False)

    rgb_cam = pipeline.createColorCamera()
    rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    stereo = get_stereo_pair(pipeline, mono_left, mono_right)

    xout_left = pipeline.createXLinkOut()
    xout_left.setStreamName('left')

    xout_right = pipeline.createXLinkOut()
    xout_right.setStreamName('right')

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName('rgb')

    xout_desp = pipeline.createXLinkOut()
    xout_desp.setStreamName('desp')

    xout_rect_left = pipeline.createXLinkOut()
    xout_rect_left.setStreamName('rect_left')

    xout_rect_right = pipeline.createXLinkOut()
    xout_rect_right.setStreamName('rect_right')

    mono_left.out.link(xout_left.input)
    mono_right.out.link(xout_right.input)
    rgb_cam.preview.link(xout_rgb.input)
    stereo.disparity.link(xout_desp.input)
    stereo.rectifiedLeft.link(xout_rect_left.input)
    stereo.rectifiedRight.link(xout_rect_right.input)



    with dai.Device(pipeline) as device:
        left_queue = device.getOutputQueue(name='left', maxSize=1)
        right_queue = device.getOutputQueue(name='right', maxSize=1)
        rgb_queue = device.getOutputQueue(name='rgb', maxSize=1)
        disparity_queue = device.getOutputQueue(name='desp', maxSize=1, blocking=False)
        rectified_left_queue = device.getOutputQueue(name='rect_left', maxSize=1, blocking=False)
        rectified_right_queue = device.getOutputQueue(name='rect_right', maxSize=1, blocking=False)
        
        disparity_multiplier = 255/stereo.initialConfig.getMaxDisparity()
        
        cv2.namedWindow("Stereo Pair")
        cv2.namedWindow("RGB")
        cv2.namedWindow("disparity")
        cv2.namedWindow("rectified Stereo Pair")
        cv2.setMouseCallback("rectified Stereo Pair", mouse_callback)



        sideBySide = False
        

        while True:
            left_frame = get_frame(left_queue)
            right_frame = get_frame(right_queue)
            rgb_frame = get_frame(rgb_queue)
            disparity = get_frame(disparity_queue)
            rect_left_frame = get_frame(rectified_left_queue)
            rect_right_frame = get_frame(rectified_right_queue)

            disparity = (disparity*disparity_multiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

            if sideBySide:
                im_out = np.hstack((left_frame, right_frame))
                rect_im_out = np.hstack((rect_left_frame, rect_right_frame))
            else:
                im_out = np.uint8(left_frame/2 + right_frame/2)
                rect_im_out = np.uint8(rect_left_frame/2 + rect_right_frame/2)
            
            rect_im_out = cv2.cvtColor(rect_im_out, cv2.COLOR_GRAY2BGR)
            rect_im_out = cv2.line(rect_im_out, (mouse_x, mouse_y),(1280, mouse_y), (0,0,128),2)
            rect_im_out = cv2.circle(rect_im_out, (mouse_x, mouse_y), 2, (255,255,128),2)

            cv2.imshow("Stereo Pair", im_out)
            cv2.imshow("RGB", rgb_frame)
            cv2.imshow("Disparity", disparity)
            cv2.imshow("rectified Stereo Pair", rect_im_out)

            


            savename = "Trial_2_Messung_1"
            counter = 0
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('t'):
                sideBySide = not sideBySide
            elif key == ord("s"):
                cv2.imwrite(savename+str(counter)+'rgb.png', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 10])
                cv2.imwrite(savename+str(counter)+'left.png', rect_left_frame, [cv2.IMWRITE_JPEG_QUALITY, 10])
                cv2.imwrite(savename+str(counter)+'right.png', rect_right_frame, [cv2.IMWRITE_JPEG_QUALITY, 10])
                cv2.imwrite(savename+str(counter)+'disparity.png', disparity, [cv2.IMWRITE_JPEG_QUALITY, 10])
                counter = counter +1

                print("saved")


