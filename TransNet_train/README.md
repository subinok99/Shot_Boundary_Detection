# Datasets   
### Download   
RAI & BBC (https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19)   
ClipShots (https://github.com/Tangshitao/ClipShots)   
(Optional) IACC.3   

### Ground Truth - format   
**consolidate_datasets.py**   
python consolidate_datasets.py   

### Generate - .tfrecord   
**create_dataset.py**   
python create_dataset.py train --mapping_fn consolidated/BBCDataset.txt --target_dir create --target_fn BBCDataset.txt   
python create_dataset.py train --mapping_fn consolidated/RAIDataset.txt --target_dir create --target_fn RAIDataset.txt   
python create_dataset.py train --mapping_fn consolidated/ClipShotsGradual.txt --target_dir create --target_fn ClipShotsGradual.txt   
python create_dataset.py train --mapping_fn consolidated/ClipShotsTrain.txt --target_dir create --target_fn ClipShotsTrain.txt   
python create_dataset.py train --mapping_fn consolidated/ClipShotsTest.txt --target_dir create --target_fn ClipShotsTest.txt   
   
# Train   
### Modify config   
**configs/transnetv2.gin**   


### Run   
python training.py configs/transnetv2.gin   

# Test   
