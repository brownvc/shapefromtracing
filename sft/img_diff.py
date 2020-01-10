import torch
import pyredner
import math
import os
import sys

img1 = sys.argv[1]
img2 = sys.argv[2]
out = sys.argv[3]

img1 = pyredner.imread(img1)
img2 = pyredner.imread(img2)

out_img = 3.0 * torch.abs(img1 - img2)
out_img = 1.0 - out_img 
pyredner.imwrite(out_img, out)