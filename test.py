'''
todo
write tests 
'''
import numpy as np
import math
import os
from PIL import Image
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
orig = os.listdir('./dataset/')
tot_pix = 0
s = 0
for i in orig:
    print(i[:-4])
    psnr_val = psnr(np.asarray(Image.open('dataset/%s'%i)),np.asarray(Image.open('out/decompressed/%scomp.png'%i[:-4])))
    img = Image.open('dataset/%s'%i)
    width, height = img.size
    pixels = width * height
    tot_pix = tot_pix + pixels
    s+=psnr_val
    print(' PSNR value is %s'%psnr_val)
avg = s/len(orig)
print('Average PSNR ratio is: %s'%avg)
orig_size = sum(os.path.getsize('dataset/%s'%f) for f in os.listdir('dataset')) 
comp_size = sum(os.path.getsize('out/compressed/%s'%f) for f in os.listdir('out/compressed/') if '.xfr' in f)
comp_ratio = orig_size/comp_size
print(orig_size)
print(comp_ratio)
print('Compression ratio is %s'%comp_ratio)
print('Original data rate is %s'%(orig_size/tot_pix))
print('compressed data rate is %s'%(comp_size/tot_pix))
'''
psnr 30.41
factor 14.25
'''