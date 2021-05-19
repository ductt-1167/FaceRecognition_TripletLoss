# Dataset:
- We using data from [Vietnamese famous people](https://drive.google.com/file/d/1kpxjaz3pIMrAhEjm7hJxcBsxKNhfl8t2/view)
but we not choose all, we filter it and save in folder data/data_training.
# Before running project, you need:
1. Install Python 3

2. Install IDE Pycharm

3. Setup interpreter: Goto Pycharm --> File --> Settings --> Project:TripletColorImage --> Python interpreter --> Add
    --> Virtualenv environment --> new environment --> Ok 
    
4. Install needed packages: choose Terminal in bottom menu
- numpy package: pip install numpy 
- pillow package: pip install pillow
- pandas package: pip install pandas 
- csv package: pip install csv 
- opencv package: pip install opencv-python 
- sklearn package: pip install sklearn 
- tensorflow package: pip install tensorflow 

# Run project
## - For training model 
- get_data.py: file get image data, reshape to input for training 
- model_with_triplet_loss: training with CNN and triplet loss function, model will save in general.model_triplet_path
-->  Customize your parameter in model_with_triplet_loss.py and general.py. Then run model_with_triplet_loss.py to train
## - For using model to register or login by face 
- register_face.py: register face user, using camera to save 50 images face and extract vector feature for this user 
- extract_feature.py: extract vector feature from the folder which save face images 
- login_face.py: identify user 
--> To register, run register_face.py 
--> To login, run login_face.py 