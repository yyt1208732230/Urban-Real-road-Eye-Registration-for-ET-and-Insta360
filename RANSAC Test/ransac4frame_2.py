import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os

frame_folder1 = "./output/ET_video_frame/"
frame_folder2 = "./output/bl_video_frame/"
outputfolder = "./output/ET_Baseline_Registra/"
outputpstfile = "./output/ET_Baseline_Registra/pst.txt"
outputdstfile = "./output/ET_Baseline_Registra/dst.txt"

def registra4eachframe(frame_folder1, frame_folder2):
    frame_count = 0
    frame_numbers = []

    # Check if the folder exists
    if not os.path.exists(frame_folder1) or not os.path.isdir(frame_folder1):
        print("Error: Folder does not exist.")
        return frame_count, frame_numbers
    
    # Get the list of files in the folder
    framefiles = os.listdir(frame_folder1)

    # for framename in framefiles:
    for index, framename in enumerate(framefiles):
        # if index <= 58:
        #     continue
        _imgname_01 = frame_folder1 + framename
        _imgname_02 = frame_folder2 + framename
        res = ransac4images(_imgname_01, _imgname_02, outputfolder, framename)
        print(res)


def ransac4images(imgname_01, imgname_02, outputfolder, newFrameName):
    sift = cv2.xfeatures2d.SIFT_create()

    img_01 = cv2.imread(imgname_01)
    img_02 = cv2.imread(imgname_02)

    img_01 = cv2.imread(imgname_01)
    img_02 = cv2.imread(imgname_02)
    print(img_01.shape)

    keypoint_01, descriptor_01 = sift.detectAndCompute(img_01, None)
    keypoint_02, descriptor_02 = sift.detectAndCompute(img_02, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(descriptor_01, descriptor_02, k = 2)
    ratio = 0.9

    good = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)  

    print(type(good))

    min_match_count = 15
    if len(good) > min_match_count:
        src_pts = np.float32([keypoint_01[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoint_02[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        #print(keypoint_01[m.queryIdx].pt for m in good)
        print(src_pts.shape)
        print(dst_pts.shape)
        
        # ransacReprojThreshold = 8.0
        ransacReprojThreshold = 9.0
        maxIters = 5000
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold, maxIters)
        #print(mask)
        print(mask.shape)

        matchesMask = mask.ravel().tolist()
        h, w, mode = img_01.shape
        pts = np.float32([[0, 0], [0, h -1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        save_arrays_to_txt(pts, outputpstfile)
        save_arrays_to_txt(dst, outputdstfile)
        print("dst")
        print(dst)
        print(dst.shape)
        img_02 = cv2.polylines(img_02, [np.int32(dst)], True, (127,255,0), 3, cv2.LINE_AA)
    else:
        print('Can not  matches!')
        matchesMask = None

    draw_params = dict(singlePointColor = None,
                    matchesMask = matchesMask,
                    flags = 2)
    img3 = cv2.drawMatches(img_01, keypoint_01, img_02, keypoint_02, good, None, **draw_params)
    
    img_ransac = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB) #RGB
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    plt.imshow(img_ransac)
    plt.savefig(outputfolder + "/" + newFrameName)

    cv2.destroyAllWindows()

    return newFrameName + " generate succeed!"

def save_arrays_to_txt(data_arrays, filename):
    try:
        # Try to load existing data from the file
        existing_data = np.loadtxt(filename, delimiter=',')
    except OSError:
        # If the file doesn't exist yet, use an empty array
        existing_data = np.array([])

    # Concatenate the existing data with the new data
    if len(data_arrays) > 0:
        new_data = np.concatenate(data_arrays, axis=0)
        all_data = np.concatenate((existing_data, new_data), axis=0)
    else:
        all_data = existing_data

    # Save the combined data to the file
    np.savetxt(filename, all_data, delimiter=',')

registra4eachframe(frame_folder1, frame_folder2)