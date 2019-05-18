import pytorch_msssim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch


def display_(x):
    if type(x) is tuple:
        img,out = x
        fig, axes = plt.subplots(ncols=1,nrows=2,figsize=(18,30))
        axes.ravel()[0].imshow(img)
        axes.ravel()[0].set_title('Original')
        axes.ravel()[1].imshow(out)
        axes.ravel()[1].set_title('After Compression')
        plt.show()
    else:
        plt.imshow(x)

def evaluate(model,ds,idx, showImages = False):
    
    x = ds[idx]
    iimg = TF.to_pil_image(x)
    x=x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    y = model(x)
    oimg = TF.to_pil_image(y.squeeze(0).cpu().detach())
    score = (pytorch_msssim.msssim(x, y).item())
#     print("MSSSIM score is {:.5f}".format(score))
    if showImages:
        display_((iimg,oimg))
    return score