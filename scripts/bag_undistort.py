#!/usr/bin/python

#this file change the imu data from perceptin to accomodate
#accelerometer and gyroscope bias

from __future__ import print_function


import rosbag
import rospy
import argparse
import cv2
from cv_bridge import CvBridgeError, CvBridge
import numpy as np
import yaml
import glob
import os


''' load_params
    Loads the parameters of the camaera from a yaml file for
    future undistortion
    @param yamlPath: absolute path to te yaml file

    @return: results of cv2.fisheye.initUndistortRectifyMap()
'''
def load_params(yamlPath):
    skip_lines = 0
    with open(yamlPath) as infile:
        for i in range(skip_lines):
            _ = infile.readline()
        data = yaml.load(infile)

    # You should replace these 3 lines with the output in calibration step
    # has been done

    # DIM = (512, 512)
    # K=np.array(YYY)
    # D=np.array(ZZZ)

    DIM = tuple(data['cam0']['resolution'])

    [fu, fv, pu, pv] = data['cam0']['intrinsics']
    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    K = np.asarray([[fu, 0, pu], [0, fv, pv], [0, 0, 1]])  # K(3,3)
    D = np.asarray(data['cam0']['distortion_coeffs'])  # D(4,1)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

    return  map1, map2


''' enhance_images
    Given an Input rosbag and Output rosbag, this function 
    will operate on the input rosbag to undistort images and
    enhance them with cv2.createCLAHE

    @param inbag:       a rosbag input file
    @param outbag:      the rosbag output file
    @param camTopics:   tuple of strings, each of which is a topic to undistort images on
    @param yamlPath:    path to the yaml file describing the intrinsics of the camera

    @return none:
'''
def enhance_images(inbag,outbag,        \
                    camTopics = ("/cam0/image_raw","/cam1/image_raw"),\
                    yamlPath = '/home/jvjohnson/Downloads/dataset-corridor4_512_16/dso/camchain.yaml'):
    rospy.loginfo(' Processing input bagfile: %s', inbag)
    rospy.loginfo(' Writing to output bagfile: %s', outbag)

    print("Processing inbag:")
    print(inbag)
    outbag = rosbag.Bag(outbag,'w',allow_unindexed='true')
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
    map1, map2 = load_params(yamlPath)

    waitingAnimation = "|/-\\"
    ind = 0
    bridge = CvBridge()
    for topic, msg, t in rosbag.Bag(inbag,'r').read_messages():
        if topic in camTopics:
            try:
                #### direct conversion to CV2 ####
                cv_image = bridge.imgmsg_to_cv2(msg, "mono8")
                #np_arr = np.fromstring(msg.data, np.uint8)
                #image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            except CvBridgeError as e:
                print("Error while converting:")
                print(e)
                return
            imShape1 = cv_image.shape
            undistorted_img = cv2.remap(cv_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            enhanced_img = clahe.apply(undistorted_img)        
            imShape2 = enhanced_img.shape
            print(waitingAnimation[int(ind%len(waitingAnimation))],end="\r")
            ind+=0.002
            newMsg = bridge.cv2_to_imgmsg(enhanced_img, "mono8")
            newMsg.header = msg.header

            outbag.write(topic,newMsg,t)
        else:
            outbag.write(topic,msg,t)
    print("Done")
    rospy.loginfo('Closing output bagfile and exiting...')
    outbag.close()




''' Main program
    Runs in the python2.7 interpreter. Usage is python bag_undistort --outbag 'outbagName' --inbag 'inbagName'

'''
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Use Image Contrast enhancement to modify exposure changes '
                                                 'for Perceptin')
    parser.add_argument('--inbag',help='input bagfile')
    parser.add_argument('--outbag',help='output bagfile')
    parser.add_argument('--yaml',help='yaml kalibr config')
    args = parser.parse_args()

    try:
        enhance_images(args.inbag,args.outbag, yamlPath=args.yaml)
    except (Exception):
        print("Encountered error while undistorting:")
        import traceback
        traceback.print_exc()
