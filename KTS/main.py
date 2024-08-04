import argparse, sys
from matplotlib import pyplot as plt
from cpd_auto import cpd_auto
import numpy as np
import h5py
from tqdm import tqdm
import glob

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def method_KTS_total(dataset, video_name, save_h5_path, max_shot):
    file = h5py.File("data/eccv16_dataset_{}_google_pool5.h5".format(dataset), 'r')   

    #plt.figure("automatic selection of the number of change-points")
    m = max_shot
    #print("max KTS : ", str(m))

    #file = h5py.File(feature_file_path, 'r')
    #print("\n file : " + h5Name)
    #print("\n vid name : " + videoName)
    data = list(file[video_name + '/features'])
    file.close()
    n_frames = len(data)
    #n_frames = file[videoName + '/n_frames'][...]
    #n_frames = 3067

    #print("\n n_frames : " + str(n_frames))
    #print("\n len(data) : " + str(len(data)))

    X = np.array(data)
    plt.plot(X)
    K = np.dot(X, X.T)
    cps, scores = cpd_auto(K, m, 1)
    print("Estimated: (m=%d)" % len(cps), cps)

    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')

    list_cps = []
    list_fps = []

    for i in range(0, len(cps) + 1):
        temp = []

        if (i == 0):  # [0, 0번째 요소]
            fir = 0
            last = cps[i]
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        elif (i == len(cps)):  # [마지막 요소, frame 수-1]
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
        #print("????")
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

    #saveLoc = resultLoc + "resultPlt.jpg"
    #plt.savefig(saveLoc)


def Use_feature_version(dataset, max_shot):
    for split_idx in range(0, 1):
        print("split_idx :", str(split_idx))


        video_list = os.listdir("data/split_video/{}/split_{}".format(dataset, split_idx))
        #save_dir = "{}/{}/{}/split{}/".format(result_path, exp_num, vid_type, split_idx)
        #video_path = "data/split_video/{}/split_{}/*.mp4".format(dataset, split_idx)
        
        for video_name in tqdm(video_list):
            #print(video_name)           
            
            video_name = video_name.split(".")[0]
            print(video_name)

            save_h5_path = "result/{}_split{}_max{}.h5".format(dataset, split_idx, max_shot)

            method_KTS_total(dataset, video_name, save_h5_path, max_shot)


def Extract_feature_version(dataset):

    return 0

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_features', default=True, help='Whether to use ground truth features or extract features from frames using Inception v3') 
    parser.add_argument('--dataset', default='summe', help='summe or tvsum')
    parser.add_argument('--max_shot', default='15', help='maximum number of shot')
    args = parser.parse_args()

    if not os.path.exists("result/"):
        os.makedirs("result/")

    use_features = args.use_features
    dataset = args.dataset
    max_shot = args.max_shot

    if (use_features):
        Use_feature_version(dataset, max_shot)
    else:
        Extract_feature_version(dataset)   
    
    

