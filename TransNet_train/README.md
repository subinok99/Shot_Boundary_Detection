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
### Edit config   
> configs/transnetv2.gin   
* datasets path  : **options.trn_files  ,  options.tst_files**
* whether you will use the 'modified model' I created  : **options.new_transnet**   
If set to False, the existing transnet v2 can be used.
* whether you will use the 'entropy similarity' I created : **TransNetV2.use_entropy_similarity**   

### Run   
> python training.py configs/transnetv2.gin   

# Test   
