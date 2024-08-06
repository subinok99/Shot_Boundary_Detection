from tensorflow.keras.preprocessing import image
import tensorflow.keras.applications.inception_v3 as inception_v3 #import InceptionV3, preprocess_input
import tensorflow.keras.applications.resnet as resnet #import ResNet50
import tensorflow.keras.applications.inception_resnet_v2 as inception_resnet_v2  #import InceptionResNetV2, preprocess_input
import tensorflow.keras.applications.vgg16 as vgg16 #import VGG16, preprocess_input
import numpy as np
import os
import natsort
import h5py
import torch


def inceptionModel(video_name, resultH5Name, frame_save_dir, data_dir):
    local_weights_file = '{}inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(data_dir)

    model = inception_v3.InceptionV3(input_shape=(100, 100, 3), weights=None, include_top=False)
    model.load_weights(local_weights_file)

    for layer in model.layers:
        layer.trainable = False

    data_list = os.listdir(frame_save_dir)

    feature = []

    total = 0
    for i in natsort.natsorted(data_list):
        img_path = frame_save_dir + "/" + i

        # 1) preprocess
        img = image.load_img(img_path, target_size=(100, 100))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # 2) feature extract
        x = inception_v3.preprocess_input(x)

        features = model.predict(x)

        flat = features.flatten()

        m = torch.nn.AvgPool1d(2, stride=2)
        flat = m(torch.Tensor([flat]))

        flat = flat.flatten()
        flat = flat.tolist()

        feature.append(flat)
        total += 1

    resultFile = h5py.File(resultH5Name, 'a')

    resultFile.create_group(video_name)
    resultFile[video_name].create_dataset("features", data = feature)
    resultFile[video_name].create_dataset("n_frames", data = total)

    resultFile.close()