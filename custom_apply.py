import os
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET

device = "cuda:1"

image_dir = "input_images"
result_dir = "test"
checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")



transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()



images_list = sorted(os.listdir(image_dir))
pbar = tqdm(total=len(images_list))
for image_name in images_list:
    print(image_name)  ### 혹시 안되는거 정확히 찾기 위해
    img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()


    a = np.array([(output_arr != 0),(output_arr != 0),(output_arr != 0)]).transpose(1,2,0)
    output_arr = a*img

    output_img = Image.fromarray(output_arr.astype("uint8"), mode="RGB")

    if image_name[-4:] == "jpeg":
        image_name = image_name[:-1] 
    
    output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))


    

