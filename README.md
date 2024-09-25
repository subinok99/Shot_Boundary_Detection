I use two models : KTS, TransNet V2

### KTS

* original project : https://github.com/TatsuyaShirakawa/KTS.git   
   
### TransNet V2
* original project : https://github.com/soCzech/TransNetV2.git   


## How to Use
**test.py** file   

Only TransNet v2   
(1) shot_extraction : Use Ground Truth (features)   
(2) shot_extraction_raw : Use Raw Video, Not Features

Only KTS   
(1) kts_shot_extraction : Use Ground Truth (features)  
(2) kts_shot_extraction_raw : Use Raw Video, Not Features

TransNetV2 & KTS   
(1) both_shot_extraction : Use Ground Truth (features)  
(2) both_shot_extraction_raw : Use Raw Video, Not Features

### Arguments
* types : Choose Function. Options : TransNet_h5, TransNet_raw , KTS_h5 , KTS_raw , Both_h5 , Both_raw
* dataset : summe or tvsum
* max_shot : (If use KTS) Maximum number of shot
* result_dir
* data_dir
* post_processing : After applying the KTS or TransNet model, Whether to apply a post-processing step. In my paper, I use this. But, If you want to test, convert to "No".