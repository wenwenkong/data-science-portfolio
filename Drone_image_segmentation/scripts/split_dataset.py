"""
---
Structure of the original folder (before splitting to train/val/test)

Data/

    images/

    masks/

---
Structure of the folder after splitting to train/val/test

Data/

    train/
        images/
        masks/

    val/
        images/
        masks/

    test/
        images/
        masks/

---
***** IMPORTANT *****:

For semantic segmentation the folder structure needs to look like below if you want to use ImageDatagenerator.
So after splitting your folders to train, val and possibly also test, rearrange them to the following format. 

Data/
    train_images/
                train/
                    img1, img2, img3, ......
    
    train_masks/
                train/
                    msk1, msk, msk3, ......
                    
    val_images/
                val/
                    img1, img2, img3, ......                

    val_masks/
                val/
                    msk1, msk, msk3, ......
      
    test_images/
                test/
                    img1, img2, img3, ......    
                    
    test_masks/
                test/
                    msk1, msk, msk3, ......
      
                
This script followed https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial121a_split_folder_into_train_test_val.ipynb

TO DO - define a simple function to make it a universal script. Let the user decide which dataset they want to split. 
"""


import splitfolders

SEED=42

input_folder = '/Users/wenwen/DataScience/Drone_image_segmentation/semantic_drone_dataset_kaggle/binary_dataset'
output_folder= '/Users/wenwen/DataScience/Drone_image_segmentation/semantic_drone_dataset_kaggle/binary_dataset_split'

splitfolders.ratio(input_folder, output=output_folder, seed=SEED, ratio=(.8, .1, .1), group_prefix=None)
