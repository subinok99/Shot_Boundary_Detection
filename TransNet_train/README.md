original project : https://github.com/soCzech/TransNetV2.git   

# Datasets   
### Download  
* RAI & BBC (https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19)   
* ClipShots (https://github.com/Tangshitao/ClipShots)   
* (Optional) IACC.3   

### Ground Truth - format   
> python **consolidate_datasets.py**   

### Generate .tfrecord   
> python **create_dataset.py** train --mapping_fn {consolidated_path} --target_dir {target_dir_path} --target_fn {target_file_name}  

   
# Train   
### How to Modify
1. **Dilation Rate & Filter Size**   
   The original TransNet V2 has 6 DDCNN v2 cells as in figure below   
   <p align="center"><img width="300" src="https://github.com/user-attachments/assets/b6464db0-b9ea-428d-b474-c47a1f70c911"></p>   

   The modified DDCNN V2 cell has two versions below.  
   (Left) When only the dilation rate is changed,
   (Right) When the dilation rate & filter size are changed.
   <p align="center"><img width="300" src="https://github.com/user-attachments/assets/38e687fa-ace6-4fa1-b763-53b4bc4b7948"><img width="300" src="https://github.com/user-attachments/assets/0a91be2c-8c10-41b6-992b-7fa2865a3fa7"></p>
   
2. **Entropy-Similarity Layer**   
In addition to the RGB histogram layer and the frame similarity layer, this is a layer that applies entropy to check the comparison between frames. This separates the R, G, and B channels for each frame, creates histograms for each, calculates the entropy value for each histogram, and then connects them into one to compare the similarity of the frames.   

The overall structural comparison can be seen in the figure below.    
(Left) Original TransNet V2,    (Right) Modified Structures.
<p align="center"><img width="700" src="https://github.com/user-attachments/assets/90cc3a0f-ce32-4a43-9b12-967def1256a6"></p>


### (Before Training) Edit config   
> configs/transnetv2.gin   
* datasets path  : **options.trn_files  ,  options.tst_files**
* whether you will use the 'modified model' I created  : **options.new_transnet**   
If set to False, the existing transnet v2 can be used.
* whether you will use the 'entropy similarity' I created : **TransNetV2.use_entropy_similarity**   

### Run   
> python training.py configs/transnetv2.gin     
   
   
# Test