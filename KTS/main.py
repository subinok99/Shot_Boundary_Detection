import argparse, sys
from matplotlib import pyplot as plt
from cpd_auto import cpd_auto
import numpy as np
import h5py
from tqdm import tqdm
import glob
import feature_extraction
import cv2
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def GT_based_KTS(dataset, video_name, save_h5_path, max_shot, data_dir):
    file = h5py.File("{}eccv16_dataset_{}_google_pool5.h5".format(data_dir, dataset), 'r')   
    data = list(file[video_name + '/features'])
    file.close()

    m = max_shot    
    n_frames = len(data)

    X = np.array(data)
    plt.plot(X)
    K = np.dot(X, X.T)
    cps, scores = cpd_auto(K, m, 1)
    #print("Estimated: (m=%d)" % len(cps), cps)

    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')

    list_cps = []
    list_fps = []

    for i in range(0, len(cps) + 1):
        temp = []

        if (i == 0):  # [0, first element]
            fir = 0
            last = cps[i]
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        elif (i == len(cps)):  # [last element, n_frames-1]
            fir = cps[i - 1] + 1
            last = n_frames - 1
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        else:
            fir = cps[i - 1] + 1
            last = cps[i]
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        list_cps.append(temp)
        list_fps.append(fps)    
    
    if (len(list_cps) <= 1):
        temp = []
        list_cps = []
        temp.append(0)
        temp.append(n_frames - 1)
        list_cps.append(temp)

    resultFile = h5py.File(save_h5_path, 'a')

    resultFile.create_group(video_name)
    resultFile[video_name].create_dataset("scenes", data=list_cps)
    resultFile[video_name].create_dataset("features", data=data)
    resultFile[video_name].create_dataset("n_frames", data=n_frames)
    resultFile.close()       

def Use_feature_version(dataset, max_shot, result_dir, data_dir):
    for split_idx in range(0, 1):
        print("split_idx :", str(split_idx))

        video_list = os.listdir("{}split_video/{}/split_{}".format(data_dir, dataset, split_idx))
        
        for video_name in tqdm(video_list, leave=True):
            video_name = video_name.split(".")[0]

            save_h5_path = "{}{}_split{}_max{}.h5".format(result_dir, dataset, split_idx, max_shot)

            GT_based_KTS(dataset, video_name, save_h5_path, max_shot, data_dir)


def toFrame(vid_path, save_dir):
    vidcap = cv2.VideoCapture(vid_path)

    if not vidcap.isOpened():
        print("Could not Open :", vid_path)
        exit(0)

    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 0
    while (vidcap.isOpened()):
        if (count >= total):
            break

        ret, image = vidcap.read()

        save_path = save_dir + "/%d.jpg" % count

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(save_path, image)
        count += 1

    vidcap.release()

def Inception_based_KTS(dataset, video_name, save_h5_path, max_shot, data_dir, video_path):
    frame_save_dir = "frame/{}/{}".format(dataset, video_name)
    toFrame(video_path, frame_save_dir)

    feature_extraction.inceptionModel(video_name, save_h5_path, frame_save_dir, data_dir)

    file = h5py.File(save_h5_path, 'r')   
    data = list(file[video_name + '/features'])
    file.close()

    m = max_shot    
    n_frames = len(data)

    X = np.array(data)
    plt.plot(X)
    K = np.dot(X, X.T)
    cps, scores = cpd_auto(K, m, 1)

    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')

    list_cps = []
    list_fps = []

    for i in range(0, len(cps) + 1):
        temp = []

        if (i == 0):  # [0, first element]
            fir = 0
            last = cps[i]
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        elif (i == len(cps)):  # [last element, n_frames-1]
            fir = cps[i - 1] + 1
            last = n_frames - 1
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        else:
            fir = cps[i - 1] + 1
            last = cps[i]
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        list_cps.append(temp)
        list_fps.append(fps)    
    
    if (len(list_cps) <= 1):
        temp = []
        list_cps = []
        temp.append(0)
        temp.append(n_frames - 1)
        list_cps.append(temp)

    resultFile = h5py.File(save_h5_path, 'a')

    resultFile[video_name].create_dataset("scenes", data=list_cps)
    resultFile.close()    

def Extract_feature_version(dataset, max_shot, result_dir, data_dir):
    for split_idx in range(0, 1):
        print("split_idx :", str(split_idx))

        video_list = os.listdir("{}split_video/{}/split_{}".format(data_dir, dataset, split_idx))
        
        #with tqdm(leave=True) as pbar:
        for video_name in video_list:
            video_path = "{}split_video/{}/split_{}/{}".format(data_dir, dataset, split_idx, video_name)            
            video_name = video_name.split(".")[0]
            save_h5_path = "{}{}_split{}_max{}.h5".format(result_dir, dataset, split_idx, max_shot)            

            Inception_based_KTS(dataset, video_name, save_h5_path, max_shot, data_dir, video_path)
            #pbar.update()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_features', default="true", help='Whether to use ground truth features or extract features from frames using Inception v3') 
    parser.add_argument('--dataset', default='summe', help='summe or tvsum')
    parser.add_argument('--max_shot', default=15, help='maximum number of shot')
    parser.add_argument('--result_dir', default="result/", help='result directory')
    parser.add_argument('--data_dir', default="data/", help='data directory')
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    use_features = args.use_features.lower()
    dataset = args.dataset
    max_shot = int(args.max_shot)

    if (use_features=="true"):
        print("\n\n--------use---------\n\n")
        Use_feature_version(dataset, max_shot, args.result_dir, args.data_dir)
    else:
        print("\n\n--------else---------\n\n")
        Extract_feature_version(dataset, max_shot, args.result_dir, args.data_dir)   
    
    

