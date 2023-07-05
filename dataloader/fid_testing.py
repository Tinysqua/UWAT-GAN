from Final_dataloader import Evaluation_dataset


from torch.utils import data
import torchvision
from torchvision import transforms
import argparse
import numpy as np
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
import sys
from scipy.linalg import sqrtm
sys.path.append("../advanced_VT/")
from networks.Gen_models import *


class Fid_Computor:
    def __init__(self, g_fine_path, g_coarse_path):
        self.model_real = torchvision.models.inception_v3(pretrained = True).cuda()
        self.model_real.eval()
        self.model_real.dropout.register_forward_hook(self.get_fea_hook_real)
        
        self.model_fake = torchvision.models.inception_v3(pretrained = True).cuda()
        self.model_fake.eval()
        self.model_fake.dropout.register_forward_hook(self.get_fea_hook_fake)
        
        self.g_model_coarse = nn.DataParallel(coarse_generator()).cuda()
        self.g_model_fine = nn.DataParallel(fine_generator()).cuda()
        self.g_model_fine.module.load_state_dict(torch.load(g_fine_path))
        self.g_model_coarse.module.load_state_dict(torch.load(g_coarse_path))
        
        self.resize_tran = transforms.Resize((299, 299))
        
        self.forward_first_warning = True
        
        self._features_real = []
        self._features_fake = []
        
    def get_fea_hook_real(self, module, input, output):
        self._features_real.append(input[0])
        
    def get_fea_hook_fake(self, module, input, output):
        self._features_fake.append(input[0])
        
    def model_forward(self, variable_lists):
        variable_lists = map(self.convert_to_cuda, variable_lists)
        X_realA, X_realB, X_realA_half, X_realB_half = variable_lists
        with torch.no_grad():
            X_fakeB_half, X_global = self.g_model_coarse(X_realA_half)
            X_fakeB = self.g_model_fine(X_realA, X_global)
            
        B_lists = [X_realB, X_fakeB, X_realB_half, X_fakeB_half]
        B_lists = map(lambda x:torch.cat([x, x, x], dim=1), B_lists)
        B_lists = map(lambda x:(x+1)/2, B_lists)
        self.X_realB, self.X_fakeB, self.X_realB_half, self.X_fakeB_half = B_lists
        self.forward_first_warning = False
        
    def extract_features(self):
        if self.forward_first_warning:
            print('This may not be using before the model_forward function')
        with torch.no_grad():
            self.model_real(self.resize_tran(self.X_realB))
            self.model_fake(self.resize_tran(self.X_fakeB))
        self.forward_first_warning = True
        
    def calculate_fid(self, act1, act2):
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        ssdiff = np.sum((mu1-mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0*covmean)
        return fid
        
    def compute(self):
        self._features_real = torch.squeeze(torch.cat(self._features_real))
        self._features_fake = torch.squeeze(torch.cat(self._features_fake))
        self._features_real = self._features_real.detach().cpu().numpy()
        self._features_fake = self._features_fake.detach().cpu().numpy()
        fid = self.calculate_fid(self._features_real, self._features_fake)
        
        return fid
        
        
    def convert_to_cuda(self, x, device=None):
        if device==None:
            return x.cuda()
        else:
            return x.to(device)
        
    def one_to_triple(X, dimension):
        return torch.cat([X, X, X], dim=dimension)
        
class Kid_Or_Fid:
    def __init__(self, g_fine_path, g_coarse_path, subset_size=50):
        self.g_model_coarse = nn.DataParallel(coarse_generator()).cuda()
        self.g_model_fine = nn.DataParallel(fine_generator()).cuda()
        self.g_model_fine.module.load_state_dict(torch.load(g_fine_path))
        self.g_model_coarse.module.load_state_dict(torch.load(g_coarse_path))
        
        self.resize_tran = transforms.Resize((299, 299)) 
        
        self.kid_model = KernelInceptionDistance(normalize=True, subset_size=subset_size).cuda()  
        self.fid_model = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
        
        self.X_realB_list = []
        self.X_fakeB_list = []
        
        self.cat_flag = False
        
    def model_forward(self, variable_lists):
        variable_lists = map(self.convert_to_cuda, variable_lists)
        X_realA, X_realB, X_realA_half, X_realB_half = variable_lists
        with torch.no_grad():
            X_fakeB_half, X_global = self.g_model_coarse(X_realA_half)
            X_fakeB = self.g_model_fine(X_realA, X_global)
            
        B_lists = [X_realB, X_fakeB, X_realB_half, X_fakeB_half]
        B_lists = map(lambda x:torch.cat([x, x, x], dim=1), B_lists)
        B_lists = map(lambda x:(x+1)/2, B_lists)
        X_realB, X_fakeB, X_realB_half, X_fakeB_half = B_lists
        self.X_realB_list.append(X_realB)
        self.X_fakeB_list.append(X_fakeB)
        
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
            return kid_mean.cpu(), kid_std.cpu()
        if compute_way == 'fid':
            self.fid_model.update(self.X_realB_list, real=True)
            self.fid_model.update(self.X_fakeB_list, real=False)
            fid_value = self.fid_model.compute()
            self.fid_model.reset()
            return fid_value.cpu()
        else:
            raise NotImplementedError('Couldn\'t find a compute way')
    
        
    def convert_to_cuda(self, x, device=None):
        if device==None:
            return x.cuda()
        else:
            return x.to(device)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--g_fine_num', type=str, default='6')
    parser.add_argument('--g_coarse_num', type=str, default='6')
    parser.add_argument('--test_data', type=str, default='dataset/')
    args = parser.parse_args()
    F_A_dataset = Evaluation_dataset(args.test_data, (1112, 1448))
    # indice = np.random.choice(len(F_A_dataset), 100)
    
    # test_dataset = data.Subset(F_A_dataset, indice)
    test_loader = data.DataLoader(F_A_dataset, batch_size=2)
    data_len = len(test_loader)
    g_fine_path = f'weights/exp{args.g_fine_num}/g_model_fine.pt'
    g_coarse_path = f'weights/exp{args.g_coarse_num}/g_model_coarse.pt'
    computer = Kid_Or_Fid(g_fine_path, g_coarse_path, subset_size=data_len)

    for variable in test_loader:
        computer.model_forward(variable)
    print('fid_result: ', computer.compute(compute_way='fid'))
    print('kid_result: ', computer.compute())
    
    
    # for variable in test_loader:
    #     fid.model_forward(variable)
    #     fid.extract_features()
      
    # final_result = fid.compute()
    # print('fid_result', final_result)
    
    
    