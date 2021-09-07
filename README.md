# human-face-compare
This repository is for comparing human face with something using domain adaptation.  
This project is launched for Unilab-2021.

### datasets
This directory consists of human images and some images which are compared with human.  
- `animal_face_datset` consists of some animal images which are collected some open source dataset.
- `human_face_datset` consists of some human images.
- `mnist` consists of some code to install mnist dataset.

If you want to add new images, you should summarize new directory which is named newly.

### OpenCV
You need OpenCV to cut off images, and you can install it from [here](https://github.com/opencv/opencv/releases). (Any version is fine.)       
We use human face detect cascade and cat face detect cascade.  
The accuracy of cascades is not perfect, but we will use these in consideration of work efficiency.  
Also, in animal case, we have only the cascade for cat face segmentation, but use this to cut out other animal images.

### preprocess.py
This code is for to create face dataset.  
Before run this code, you need to make `original` directory in dataset directory and put the image you want to process here.  
```
dataset
 |-- animal_face_dataset  
 |            |-- original <-- here
 |
 |-- human_face_dataset  
              |-- original <-- here
```  
Firstly, you use `human_face_segmentation_preprocess` or `animal_face_segmentation_preprocess` function for to create segmented images.  
The segment images are saved in `segment_image directory` by original image directory.  
After that, User should check images which are cut out correctly and organize `segment_image directory`.   
Finally, you use `resize_segmented_images` function for to resize segment_images.  
- If you want to check resized images, you use `demonstration` function.  
- If you want to rename image files, you use `rename_image_file` function. However, you must not run twice in a row because some image files are overwritten by the other image files.  

### adda
This directory is for training adversarial domain adaptation model.  
You can train model without gpu in the default condition, and you should edit code if you want to train with large dataset.  