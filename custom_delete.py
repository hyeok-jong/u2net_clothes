import os  
import cv2  
import numpy as np

list_dir = os.listdir("input_images")
os.chdir('./input_images')

cnt = 0

for i in list_dir:
    if i[-4:] == "webp" or i[-4:] == ".gif":  # delete html and gif
        os.remove(i)
        cnt += 1
print(f"deleted {cnt} images due to format")

cnt = 0
list_dir = os.listdir(os.getcwd())
for i in list_dir:
    image_np = cv2.imread(i)
    if image_np is None:
        os.remove(i)
        image_np = np.empty([2,2,3])
    image_np = image_np.shape
    
    if image_np[0]*image_np[1] > 2000000:
        os.remove(i)
        cnt += 1
print(f"deleted {cnt} images due to size")



