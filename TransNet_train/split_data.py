'''
consolidated의 Train을 Train(80%)/ Validation(20%)으로 split하는 코드

'''


import random
import os
import shutil

'''
path = "consolidated/ClipShotsTrain_Train/"
file_list = os.listdir(path)
file_list_txt = [file for file in file_list if file.endswith(".txt")]
txt_file_name = "consolidated/ClipShotsTrain_Train.txt"

plus_name = "D:\Video_Summarization\TransNet_train\\raw_data\ClipShots\\videos\\train\\"
plus_name2 = "D:\Video_Summarization\TransNet_train\consolidated\ClipShotsTrain_Train\\"
'''

path = "consolidated/ClipShotsTrain_Valid/"
file_list = os.listdir(path)
file_list_txt = [file for file in file_list if file.endswith(".txt")]
txt_file_name = "consolidated/ClipShotsTrain_Valid.txt"

plus_name = "D:\Video_Summarization\TransNet_train\\raw_data\ClipShots\\videos\\train\\"
plus_name2 = "D:\Video_Summarization\TransNet_train\consolidated\ClipShotsTrain_Valid\\"

f= open(txt_file_name,"w+")
for i in file_list_txt:
    png_name = i.split(".")[0] + ".mp4"

    f.write( plus_name + png_name + "," + plus_name2 + str(i)+"\n")

f.close()



'''
path = "consolidated/ClipShotsTrain/"
file_list = os.listdir(path)
file_list_txt = [file for file in file_list if file.endswith(".txt")]

len_txt = len(file_list_txt)
train_ratio = int(len_txt * 0.8)

idx = 0
file_list = []

for i in range(train_ratio):
    a = random.randint(0, len_txt-1)

    while a in file_list:
        a = random.randint(0, len_txt-1)
    
    file_list.append(a)

    from_file_path = "consolidated/ClipShotsTrain/" + file_list_txt[a] # 복사할 파일
    to_file_path =  "consolidated/ClipShotsTrain_Train/" + file_list_txt[a] # 복사 위치 및 파일 이름 지정
    shutil.move(from_file_path, to_file_path) 

    png_name = file_list_txt[a].split(".")[0] + ".png"

    from_file_path = "consolidated/ClipShotsTrain/" +  png_name# 복사할 파일
    to_file_path =  "consolidated/ClipShotsTrain_Train/" + png_name # 복사 위치 및 파일 이름 지정
    shutil.move(from_file_path, to_file_path) 
'''