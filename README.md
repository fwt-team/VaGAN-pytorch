# VaGAN-Pytorch
---

### File
datasets  # container of data  
ganvade # core code  
runs # the trained model   

---
### Requirements
matplotlib==3.1.1  
numpy==1.16.4  
pandas==0.23.2  
scipy==1.1.0  
sklearn==0.21.3  
torch==1.2.0  
torchvision==0.4.0   
tqdm==4.35.0  

---
### Run VaGAN
__params:__  
-r Name of training run  
-n Number of epochs  
-s Name of dataset, default is minist  
-v version of model  
-ns control the switch of noise  

__example:__  
python train.py -r vagan -n 500 -s mnist -v 1 -ns 0  

