# Clothe segmentation  

## 0.Refer   
[Original repository](https://github.com/levindabhi/cloth-segmentation)  

## 1. Trained model  

First download trained model from original repository.  
Then move to `trained_checkpoint/`    

## 2. Set images  

Put images to be segmented in folder `input_images`    

## 3. Preprocessing.  

Beacause I used this model with crawled data, some of them are `.html` or `.gif` and rarely size of image is too big for model even though I used GPU which has 24GB memory.  

So by running `custom_apply.py` one can delete such formats and BBBIG images.  
(Note that function between image size and file size is not monotone, and for model image size(w,h) is key. Not filesize.)  

## 4. Test.  

Running `custom_apply.py` one can get only clothe images.  


