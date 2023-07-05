import sys
sys.path.append("../diy_VT/")
from dataloader.Final_dataloader import Evaluation_dataset
from utils.visualization import Patch_img_generator
from torch.utils import data

import torch
from torchvision import transforms
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from networks.Gen_models import *

class Kid_Or_Fid:
    def __init__(self, original_changer, original_img_size, img_size, if_cuda=True):
        self.g_model_coarse = None
        self.g_model_fine = None
        F_A_dataset = Evaluation_dataset("dataset/", original_img_size=original_img_size, img_size=img_size)
        self.test_loader = data.DataLoader(F_A_dataset, batch_size=1)
        subset_size = len(self.test_loader)
        self.patch_generator = Patch_img_generator(original_changer=original_changer)
        
        self.resize_tran = transforms.Resize((299, 299)) 
        self.if_cuda = if_cuda
        if if_cuda:
            self.kid_model = KernelInceptionDistance(normalize=True, subset_size=subset_size).cuda()  
            self.fid_model = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
        else:
            self.kid_model = KernelInceptionDistance(normalize=True, subset_size=subset_size)  
            self.fid_model = FrechetInceptionDistance(feature=2048, normalize=True)   
        
        self.X_realB_list = []
        self.X_fakeB_list = []
        
        self.cat_flag = False
        
    def spin_once(self):
        for variable in self.test_loader:
            self.model_forward(variable)
        fid_score = self.compute(compute_way='fid')
        kid_score = self.compute()
        self.X_realB_list = []
        self.X_fakeB_list = []
        self.cat_flag = False
        return fid_score.item(), kid_score[0].item(), kid_score[1].item()
        
    def model_forward(self, variable_lists):
        variable_lists = map(self.convert_to_cuda, variable_lists)
        X_realA, X_realB, X_realA_half, X_realB_half = variable_lists
        X_fakeB_half, X_fakeB = self.patch_generator.generate(X_realA, X_realA_half, g_model_coarse=self.g_model_coarse,
                                                              g_model_fine=self.g_model_fine)
        # with torch.no_grad():
        #     X_fakeB_half, X_global = self.g_model_coarse(X_realA_half)
        #     X_fakeB = self.g_model_fine(X_realA, X_global)
            
        B_lists = [X_realB, X_fakeB, X_realB_half, X_fakeB_half]
        B_lists = map(lambda x:torch.cat([x, x, x], dim=1), B_lists)
        B_lists = map(lambda x:(x+1)/2, B_lists)
        X_realB, X_fakeB, X_realB_half, X_fakeB_half = B_lists
        self.X_realB_list.append(X_realB if self.if_cuda else X_realB.cpu())
        self.X_fakeB_list.append(X_fakeB if self.if_cuda else X_fakeB.cpu())
        
    def compute(self, compute_way = 'kid'):
        if not self.cat_flag:
            self.X_realB_list = self.resize_tran(torch.cat(self.X_realB_list))
            self.X_fakeB_list = self.resize_tran(torch.cat(self.X_fakeB_list))
            self.cat_flag = True
        if compute_way == 'kid':
            self.kid_model.update(self.X_realB_list, real=True)
            self.kid_model.update(self.X_fakeB_list, real=False)
            kid_mean, kid_std = self.kid_model.compute()
            self.kid_model.reset()
            return (kid_mean.cpu(), kid_std.cpu()) if self.if_cuda else (kid_mean, kid_std)
        elif compute_way == 'fid':
            self.fid_model.update(self.X_realB_list, real=True)
            self.fid_model.update(self.X_fakeB_list, real=False)
            fid_value = self.fid_model.compute()
            self.fid_model.reset()
            return fid_value.cpu() if self.if_cuda else fid_value
        else:
            raise NotImplementedError('Couldn\'t find a compute way')
    
        
    def convert_to_cuda(self, x, device=None):
        if device==None:
            return x.cuda()
        else:
            return x.to(device)
    
    def update_models(self, g_fine_model, g_coarse_model):
        self.g_model_fine = g_fine_model
        self.g_model_coarse = g_coarse_model
        
        
if __name__ == "__main__":
    original_changer = transforms.Resize((2432, 3072))
    original_img_size = (2432, 3072)
    img_size = (608, 768)
    metrics_computer = Kid_Or_Fid(original_changer=original_changer, original_img_size=original_img_size,
                                   img_size=img_size, if_cuda=False)
    g_model_coarse = Global_Generator(input_nc=3, output_nc=1).cuda()
    g_model_fine = Local_Enhancer(input_nc=3, output_nc=1, img_sz=256).cuda()
    metrics_computer.update_models(g_fine_model=g_model_fine, g_coarse_model=g_model_coarse)
    print(metrics_computer.spin_once())
  