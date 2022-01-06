# VaGAN-Pytorch
Paper: Clustering Analysis via Deep Generative Models With Mixture Models
---

### File
    datasets  # container of data  
    vagan     # core code  
    runs      # the trained model   

---
### Requirements
    matplotlib  ==3.1.1  
    numpy       ==1.16.4  
    pandas      ==0.23.2  
    scipy       ==1.1.0  
    sklearn     ==0.21.3  
    torch       ==1.2.0  
    torchvision ==0.4.0   
    tqdm        ==4.35.0  

---
### Run VaGAN
__params:__  

    -r   #Name of training run  
    -n   #Number of epochs  
    -s   #Name of dataset, default is minist  
    -v   #version of model  
    -ns  #control the switch of noise  

__example（GMM）:__  
python train.py -r vagan -n 500 -s mnist -v 1 -ns 0  
__example（SMM）:__  
python train_smm_v2.py -r vagan -n 500 -s mnist -v 2 -ns 0  

### Result
__trained model:__  
https://drive.google.com/drive/folders/1PO_uRSCBJn6t4TVkjSo9_x_QTxtanujl?usp=sharing

---
### Reference
If you use our code in your work, please cite our paper. 

    @ARTICLE{YANG2020,  
    author={Yang, Lin and Fan, Wentao and Bouguila, Nizar},  
    journal={IEEE Transactions on Neural Networks and Learning Systems},   
    title={Clustering Analysis via Deep Generative Models With Mixture Models},   
    year={2022},  
    volume={33},  
    number={1},  
    pages={340-350},  
    doi={10.1109/TNNLS.2020.3027761}}

