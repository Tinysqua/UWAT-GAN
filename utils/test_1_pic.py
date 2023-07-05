from torchvision import transforms
from PIL import Image
import sys
sys.path.append("../advanced_VT/")
from torch import nn
from models.models import *
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from visdom import Visdom
viz_image = Visdom()

def convert_to_cuda(x, device=None):
    if device==None:
        return x.cuda()
    else:
        return x.to(device)
    
    
BIGGER_SIZE = (832*2, 1088*2)
SMALLER_SIZE = (832, 1088)

def one_to_triple(X, dimension):
    return torch.cat([X, X, X], dim=dimension)

transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize(BIGGER_SIZE),
                                  transforms.Normalize(0.5, 0.5)])

transformer_resize = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize(SMALLER_SIZE),
                                  transforms.Normalize(0.5, 0.5)])


g_model_coarse = nn.DataParallel(coarse_generator()).cuda()
g_model_fine = nn.DataParallel(fine_generator()).cuda()


g_model_fine.module.load_state_dict(torch.load('weights/exp2/g_model_fine.pt'))
g_model_coarse.module.load_state_dict(torch.load('weights/exp2/g_model_coarse.pt'))



def funloader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
def angloader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
    
fun = funloader('/home/fzj/advanced_VT/dataset/data/Images/3_7.png')
an = angloader('/home/fzj/advanced_VT/dataset/data/Masks/3_mask_7.png')



X_realA = transformer(fun).unsqueeze(0)
X_realB = transformer(an).unsqueeze(0)

X_realA_half = transformer_resize(fun).unsqueeze(0)
X_realB_half = transformer_resize(an).unsqueeze(0)

variable_list = [X_realA, X_realB, X_realA_half, X_realB_half]
variable_list = map(convert_to_cuda, variable_list)


X_realA, X_realB, X_realA_half, X_realB_half = variable_list

with torch.no_grad():
    X_fakeB_half, X_global = g_model_coarse(X_realA_half)
    X_fakeB = g_model_fine(X_realA, X_global)

X_realB = one_to_triple(X_realB, dimension=1)
X_fakeB = one_to_triple(X_fakeB, dimension=1)
X_realB_half = one_to_triple(X_realB_half, dimension=1)
X_fakeB_half = one_to_triple(X_fakeB_half, dimension=1)

display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0).cpu().detach()
display_list = (display_list + 1) / 2

viz_image.images(display_list, env="eval_VT_global", opts=dict(title= 'evaluation'), nrow=1)

display_list = torch.cat([X_realA, X_fakeB, X_realB], dim=0).cpu().detach()
display_list = (display_list + 1) / 2

viz_image.images(display_list, env="eval_VT_local", opts=dict(title= 'evaluation' ), nrow=1)








    