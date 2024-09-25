from TransNet import TransNetV2
import h5py
import os
import tensorflow as tf
import numpy as np
import PIL
import kts
import cv2
from skimage.metrics import structural_similarity as ssim
import argparse
from matplotlib import pyplot as plt
from cpd_auto import cpd_auto
import feature_extraction

# only TransNetV2 -- use .h5 (features)
def shot_extraction(data_dir, dataset, video_name, weight_file, save_h5_path):
    file = h5py.File("{}eccv16_dataset_{}_google_pool5.h5".format(data_dir, dataset), 'r')   
    data = list(file[video_name + '/features'])
    file.close()
 
    n_frames = len(data)

    model = TransNetV2(weight_file)

    vid_name = video_name

    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(file)
    scenes = model.predictions_to_scenes(single_frame_predictions)

    resultFile = h5py.File(save_h5_path, 'a')

    resultFile.create_group(video_name)
    resultFile[vid_name].create_dataset("scenes", data=scenes)
    resultFile[vid_name].create_dataset("features", data=data)
    resultFile[vid_name].create_dataset("n_frames", data=n_frames)

    resultFile.close()

# only TransNetV2 -- raw video
def shot_extraction_raw(video_file, weight_file, save_h5_path):
    model = TransNetV2(weight_file)

    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_file)
    scenes = model.predictions_to_scenes(single_frame_predictions)

    resultFile = h5py.File(save_h5_path, 'a')

    resultFile[video_name].create_dataset("scenes", data=scenes)

    resultFile.close()


# only KTS -- use .h5 (features)
def kts_shot_extraction(data_dir, datasets, video_name, save_h5_path, max_shot):
    file = h5py.File("{}eccv16_dataset_{}_google_pool5.h5".format(data_dir, datasets), 'r')   
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

    resultFile.create_group(video_name)
    resultFile[video_name].create_dataset("scenes", data=list_cps)
    resultFile[video_name].create_dataset("features", data=data)
    resultFile[video_name].create_dataset("n_frames", data=n_frames)
    resultFile.close()  

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

# only KTS -- raw video
def kts_shot_extraction_raw(dataset, video_name, save_h5_path, max_shot, data_dir, video_path):
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


def both_shot_extraction(save_h5_path, vid_name, transNet_scenes, frame_save_dir, total):
    _, _, kts_scenes = kts.method_KTS_total(save_h5_path, vid_name, total)

   # print(kts_scenes)
   # print("\n\n---\n\n")
   # print(transNet_scenes)
   # print(kts_scenes)
   # print(vid_name)

    kts_scene_list = [0 for _ in range(kts_scenes[-1][1])]
    trans_scene_list = [0 for _ in range(transNet_scenes[-1][1])]

    for x in range(len(kts_scene_list)):
        for y in range(len(kts_scenes)):
            if (x == kts_scenes[y][1]):
                kts_scene_list[x] = 1

    for x in range(len(trans_scene_list)):
        for y in range(len(transNet_scenes)):
            if (x == transNet_scenes[y][1]):
                trans_scene_list[x] = 1

    # or
    new_scenes = []
    #print("\n kts len : " + str(len(kts_scene_list)))
    #print("\n trans len : " + str(len(trans_scene_list)))
    idx = len(kts_scene_list) if (len(kts_scene_list) < len(trans_scene_list)) else len(trans_scene_list)
    for x in range(idx):
        if (kts_scene_list[x] or trans_scene_list[x]):
            new_scenes.append(1)
        else:
            new_scenes.append(0)


    # interval < 20 --> 하나로 연결
    prev = 0
    interval = 20
    count = 0
    new_idx= 0

    frame_dir = frame_save_dir + "/"
    for x in range(len(new_scenes)):
        if (new_scenes[x] == 1):
            if (count <= interval):
                if (prev != 0):
                    # SSIM 양옆

                    new_scenes[prev] = 0
                    new_scenes[x] = 0

                    b1_for = cv2.imread(frame_dir + str(prev+1) + ".jpg")
                    b1_back = cv2.imread(frame_dir + str(prev-1) + ".jpg")
                    b1 = cv2.imread(frame_dir + str(prev) + ".jpg")
                    b2_for = cv2.imread(frame_dir + str(x-1) + ".jpg")
                    b2_back = cv2.imread(frame_dir + str(x+1) + ".jpg")
                    b2 = cv2.imread(frame_dir + str(x) + ".jpg")

                    (b1_for_score, diff) = ssim(b1, b1_for, full=True)
                    (b2_for_score, diff) = ssim(b2, b2_for, full=True)
                    (b1_back_score, diff) = ssim(b1, b1_back, full=True)
                    (b2_back_score, diff) = ssim(b2, b2_back, full=True)

                    b1_score = b1_for_score + b1_back_score
                    b2_score = b2_for_score + b2_back_score

                    if (b1_score < b2_score):
                        new_scenes[prev] = 1
                    else:
                        new_scenes[x] = 1


                    # SSIM center

                    # cosine 양옆
                    '''
                    new_scenes[prev] = 0
                    new_scenes[x] = 0

                    b1_for = np.array(cv2.imread(frame_dir + str(prev + 1) + ".jpg")).flatten() / 255.
                    b1_back = np.array(cv2.imread(frame_dir + str(prev - 1) + ".jpg")).flatten()/ 255.
                    b1 = np.array(cv2.imread(frame_dir + str(prev) + ".jpg")).flatten()/ 255.
                    b2_for = np.array(cv2.imread(frame_dir + str(x - 1) + ".jpg")).flatten()/ 255.
                    b2_back = np.array(cv2.imread(frame_dir + str(x + 1) + ".jpg")).flatten()/ 255.
                    b2 = np.array(cv2.imread(frame_dir + str(x) + ".jpg")).flatten()/ 255.

                    b1_for_score = spatial.distance.cosine(b1, b1_for)
                    b2_for_score = spatial.distance.cosine(b2, b2_for)
                    b1_back_score = spatial.distance.cosine(b1, b1_back)
                    b2_back_score = spatial.distance.cosine(b2, b2_back)

                    b1_score = b1_for_score + b1_back_score
                    b2_score = b2_for_score + b2_back_score

                    if (b1_score > b2_score):
                        new_scenes[prev] = 1
                    else:
                        new_scenes[x] = 1
                    '''

                    # cosine center

                    # wasserstein distance

                    '''
                    new_scenes[prev] = 0
                    new_scenes[x] = 0

                    b1_for = cv2.imread(frame_dir + str(prev + 1) + ".jpg")
                    b1_back = cv2.imread(frame_dir + str(prev - 1) + ".jpg")
                    b1 = cv2.imread(frame_dir + str(prev) + ".jpg")
                    b2_for = cv2.imread(frame_dir + str(x - 1) + ".jpg")
                    b2_back = cv2.imread(frame_dir + str(x + 1) + ".jpg")
                    b2 = cv2.imread(frame_dir + str(x) + ".jpg")

                    (b1_for_score, diff) = wasserstein_distance(b1, b1_for)
                    (b2_for_score, diff) = wasserstein_distance(b2, b2_for)
                    (b1_back_score, diff) = wasserstein_distance(b1, b1_back)
                    (b2_back_score, diff) = wasserstein_distance(b2, b2_back)

                    b1_score = b1_for_score + b1_back_score
                    b2_score = b2_for_score + b2_back_score

                    if (b1_score < b2_score):
                        new_scenes[prev] = 1
                    else:
                        new_scenes[x] = 1
                    '''

                    # gromov wasserstein distance

                    # 단순 center
                    '''                    
                    new_scenes[prev] = 0
                    new_scenes[x] = 0

                    new_idx = (prev + x) * 0.5
                    new_scenes[int(new_idx)] = 1
                    '''

                count = 0
            count += 1
            prev = new_idx

    #print(new_scenes)

    result = []
    fir = 0
    sec = 0
    for i in range(len(new_scenes)):
        temp = []
        if (new_scenes[i] == 1):
            temp.append(fir)
            temp.append(sec)
            fir = (i + 1)
            result.append(temp)
        if (i == (len(new_scenes) - 1)):
            temp.append(fir)
            temp.append(i)
            result.append(temp)
        sec += 1

    # shot < 20 --> 없애기

    return result


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--types', default="TransNet_h5", help='TransNet_h5 or TransNet_raw or KTS_h5 or KTS_raw or Both_h5 or Both_raw') 
    parser.add_argument('--dataset', default='summe', help='summe or tvsum')
    parser.add_argument('--max_shot', default=15, help='(If use KTS) Maximum number of shot')
    parser.add_argument('--result_dir', default="result/", help='Result Directory')
    parser.add_argument('--data_dir', default="data/", help='Data Directory')
    #### parser.add_argument('--post_processing', default="Yes", help='After applying the KTS or TransNet model, Whether to apply a post-processing step')
    args = parser.parse_args()

    types = args.types.lower()
    datasets = args.dataset.lower()

    if (types == "transnet_h5"):
        for split_idx in range(0, 4):
            print("split_idx :", str(split_idx))

            video_list = os.listdir("{}split_video/{}/split_{}".format(args.data_dir, datasets, split_idx))
            
            for video_name in video_list:
                video_name = video_name.split(".")[0]

                save_h5_path = "{}{}/{}_split{}.h5".format(args.result_dir, types, datasets, split_idx)

                shot_extraction(args.data_dir, datasets, video_name, args.weight_file, args.result_dir)


    elif (types == "transnet_raw"):
        for split_idx in range(0, 4):
            print("split_idx :", str(split_idx))

            video_list = os.listdir("{}split_video/{}/split_{}".format(args.data_dir, datasets, split_idx))
            
            for video_name in video_list:
                video_path = "{}split_video/{}/split_{}/{}.mp4".format(args.data_dir, datasets, split_idx, video_name)
                save_h5_path = "{}{}/{}_split{}.h5".format(args.result_dir, types, datasets, split_idx)

                shot_extraction_raw(video_path, args.weight_file, save_h5_path)


    elif (types == "kts_h5"):
        for split_idx in range(0, 4):
            print("split_idx :", str(split_idx))

            video_list = os.listdir("{}split_video/{}/split_{}".format(args.data_dir, datasets, split_idx))
            
            for video_name in video_list:
                video_name = video_name.split(".")[0]

                save_h5_path = "{}{}/{}_split{}_kts{}.h5".format(args.result_dir, types, datasets, split_idx, args.max_shot)

                kts_shot_extraction(args.data_dir, datasets, video_name, save_h5_path, args.max_shot)


    elif (types == "kts_raw"):
        for split_idx in range(0, 4):
            print("split_idx :", str(split_idx))

            video_list = os.listdir("{}split_video/{}/split_{}".format(args.data_dir, datasets, split_idx))
            
            for video_name in video_list:
                video_path = "{}split_video/{}/split_{}/{}.mp4".format(args.data_dir, datasets, split_idx, video_name)
                save_h5_path = "{}{}/{}_split{}.h5".format(args.result_dir, types, datasets, split_idx)

                kts_shot_extraction_raw(datasets, video_name, save_h5_path, args.max_shot, args.data_dir, video_path)
        
        
    elif (types == "both_h5"):
        for split_idx in range(0, 4):
            print("split_idx :", str(split_idx))

            video_list = os.listdir("{}split_video/{}/split_{}".format(args.data_dir, datasets, split_idx))
            
            for video_name in video_list:
                video_name = video_name.split(".")[0]

                save_h5_path = "{}{}/{}_split{}_kts{}.h5".format(args.result_dir, types, datasets, split_idx, args.max_shot)

                both_shot_extraction()
    
    else:
        for split_idx in range(0, 4):
            print("split_idx :", str(split_idx))

            video_list = os.listdir("{}split_video/{}/split_{}".format(args.data_dir, datasets, split_idx))
            
            for video_name in video_list:
                video_path = "{}split_video/{}/split_{}/{}.mp4".format(args.data_dir, datasets, split_idx, video_name)
                save_h5_path = "{}{}/{}_split{}.h5".format(args.result_dir, types, datasets, split_idx)

                both_shot_extraction_raw()