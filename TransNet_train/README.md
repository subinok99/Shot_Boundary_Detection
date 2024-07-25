# Datasets   
### Download  
* RAI & BBC (https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19)   
* ClipShots (https://github.com/Tangshitao/ClipShots)   
* (Optional) IACC.3   

### Ground Truth - format   
**consolidate_datasets.py**   
> python consolidate_datasets.py   

### Generate .tfrecord   
**create_dataset.py**   
> python create_dataset.py train --mapping_fn {consolidated_path} --target_dir {target_dir_path} --target_fn {target_file_name}  

   
# Train   
### Modify config   
**configs/transnetv2.gin**   
* datasets path
  > options.trn_files   
  > options.tst_files 

### Run   
> python training.py configs/transnetv2.gin   

# Test   
