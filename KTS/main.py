from matplotlib import pyplot as plt
from cpd_auto import cpd_auto
import numpy as np
import h5py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


original_h5_path = "original"

for split_idx in range(2, 3):
    print("\n------------splitidx : {}".format(split_idx))

    #if not os.path.exists("{}/{}/{}/split{}".format(result_path, exp_num, vid_type, split_idx)):
    #    os.makedirs("{}/{}/{}/split{}/".format(result_path, exp_num, vid_type, split_idx))

    video_list = os.listdir("split_video/sumMe/split_{}".format(split_idx))
    #save_dir = "{}/{}/{}/split{}/".format(result_path, exp_num, vid_type, split_idx)


    # 1. inference
    for video_name in video_list:
        video_name = video_name.split(".")[0]
        print(video_name)
        save_h5_path = "{}_{}.h5".format(original_h5_path, split_idx)

        file = h5py.File(save_h5_path, 'r')
        data = list(file[video_name + '/features'])
        total = len(data)
        file.close()

        _, _, scenes = kts.method_KTS_total(save_h5_path, video_name, total)

        if (len(scenes) <= 1):
            #print("????")
            temp = []
            final_scenes = []
            temp.append(0)
            temp.append(total - 1)
            final_scenes.append(temp)

        new_save_h5_path = "{}_{}_kts15.h5".format(original_h5_path, split_idx)
        resultFile = h5py.File(new_save_h5_path, 'a')

        resultFile.create_group(video_name)
        resultFile[video_name].create_dataset("scenes", data=scenes)
        resultFile[video_name].create_dataset("features", data=data)
        resultFile[video_name].create_dataset("n_frames", data=total)
        resultFile.close()

