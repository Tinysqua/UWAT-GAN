from visdom import Visdom
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import sys
sys.path.append('../diy_VT/')
from networks.Gen_models import *
from dataloader.VT_dataloader import slo_ffa_dataset
import torch
from torchvision import transforms
from torchvision import utils

def convert_to_cuda(x, device=None):
    if device==None:
        return x.cuda()
    else:
        return x.to(device)
    
    
def one_to_triple(X, dimension=1):
    return torch.cat([X, X, X], dim=dimension)

def unnormalize(X):
    return (X + 1)/2
    
    
class Visualizer:
    def __init__(self, original_changer, weights_up_dir, way='tensorboard'):
        self.patch_generator = Patch_img_generator(original_changer=original_changer)
        if way == 'tensorboard':
            self.recorder = SummaryWriter(weights_up_dir)
            self.use_tensorboard = True
        elif way == 'visdom':
            self.recorder = Visdom()
            self.use_tensorboard = False
        else:
            raise NotImplementedError('Visulizer [%s] is not implemented' % way)
        
    def scalars_initialize(self):
        if not self.use_tensorboard:
            self.recorder.line([[5.], [5.], [70.]], [0], win="VTGAN_LOSS", opts=dict(title='loss',
                                                                       legend=['d_f_loss', 'd_c_loss', 'gan_loss']))
            self.recorder.line([[70.]], [0], win="Fid_score", opts=dict(title='Fid',
                                                                       legend=['fid']))
            self.recorder.line([[0.], [0.]], [0], win="Kid_score", opts=dict(title='Kid',
                                                                   legend=['kid_mean', 'kid_std']))
            
    def scalar_adjuster(self, values, step, title, legend=None):
        if self.use_tensorboard:
            self.tb_draw_scalars(values, step, title, legend)
        else:
            self.viz_draw_scalars(values, step, title)
            
    def tb_draw_scalars(self, values, step, title, legend):
        self.recorder.add_scalars(main_tag=title, 
                                  tag_scalar_dict=dict(zip(legend, values)), global_step=step)
        self.recorder.flush()
        
            
    def viz_draw_scalars(self, values, step, title):
        value_len = len(values)
        visdom_list = []
        for i in range(value_len):
            visdom_list.append([values[i]])
        self.recorder.line(visdom_list, step, win=title, update='append')
        
        
    # draw images per epoch    
    def iter_summarize_performance(self, g_f_model, g_c_model, iter_thing, iteration_str):
        X_realA, X_realB, X_realA_half, X_realB_half = next(iter_thing)
        env_tag = ("VT_global", "VT_local")
        
        X_realA = convert_to_cuda(X_realA)
        X_realB = convert_to_cuda(X_realB)
        X_realA_half = convert_to_cuda(X_realA_half)
        X_realB_half = convert_to_cuda(X_realB_half)
        
        X_fakeB_half, X_fakeB = self.patch_generator.generate(X_realA, X_realA_half, g_c_model, g_f_model)
        
        
        X_realB = one_to_triple(X_realB, dimension=1)
        X_fakeB = one_to_triple(X_fakeB, dimension=1)
        X_realB_half = one_to_triple(X_realB_half, dimension=1)
        X_fakeB_half = one_to_triple(X_fakeB_half, dimension=1)
        
        display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half], dim=0).cpu().detach()
        display_list = (display_list + 1) / 2
        
        if self.use_tensorboard:
            self.recorder.add_images(env_tag[0], display_list, iteration_str) 
            self.recorder.flush()
        else:
            self.recorder.images(display_list, env=env_tag[0], opts=dict(title= iteration_str), nrow=1)
        
        display_list = torch.cat([X_realA, X_fakeB, X_realB], dim=0).cpu().detach()
        display_list = (display_list + 1) / 2
        
        if self.use_tensorboard:
            self.recorder.add_images(env_tag[1], display_list, iteration_str)
            self.recorder.flush() 
        else:
            self.recorder.images(display_list, env=env_tag[1], opts=dict(title= iteration_str), nrow=1)
            
    def close_recorder(self):
        if self.use_tensorboard:
            self.recorder.close()
            
class Pic_saver:
    def __init__(self, original_changer,  Gen_C, Gen_F, patch_img_size=256):
        self.patch_img_generator = Patch_img_generator(original_changer,patch_img_size)
        self.g_fine = Gen_F
        self.g_coarse = Gen_C
        
    def save_img(self, variable_list, index, Coarse_save, Fine_save):
        variable_list = map(convert_to_cuda, variable_list)
        X_realA_original, X_realB, X_realA_half, X_realB_half = variable_list
        
        X_fakeB_half, X_fakeB  = self.patch_img_generator.generate(X_realA_original, X_realA_half, self.g_coarse, self.g_fine)
        A_list = [X_realA_original, X_realA_half]
        B_list = [X_realB, X_fakeB, X_realB_half, X_fakeB_half]
        B_list = map(one_to_triple, B_list)
        B_list = map(unnormalize, B_list)
        A_list = map(unnormalize, A_list)
        X_realA_original, X_realA_half = A_list
        X_realB, X_fakeB, X_realB_half, X_fakeB_half = B_list
        
        
        display_list = torch.cat([X_realA_half, X_fakeB_half, X_realB_half])
        utils.save_image(display_list, f'{Coarse_save}{index}.png')
        display_list = torch.cat([X_realA_original, X_fakeB, X_realB])
        utils.save_image(display_list, f'{Fine_save}{index}.png')
  
            
class Patch_img_generator:
    def __init__(self, original_changer, patch_img_size=256):
        self.patch_img_size = patch_img_size
        self.original_changer = original_changer
        
    
    def tensor_chunk_advanced(self, tensor):
        self.bs = tensor.shape[0]
        # one site split
        splited_tensor = torch.split(tensor, self.patch_img_size, dim=3)
        cated_tensor = torch.cat(splited_tensor, dim=0)
        # next site split
        splited_tensor = torch.split(cated_tensor, self.patch_img_size, dim=2)
        return splited_tensor

    def tensor_cat_advanced(self, splited_tensor):
        cated_tensor = torch.cat(splited_tensor, dim=2)
        splited_tensor = torch.split(cated_tensor, self.bs, dim=0)
        cated_tensor = torch.cat(splited_tensor, dim=3)
        return cated_tensor
    
    def generate(self, X_realA_original, X_realA_half, 
                 g_model_coarse, g_model_fine):
        with torch.no_grad():
            X_fakeB_half, x_global = g_model_coarse(X_realA_half)
        XA_original_splited = self.tensor_chunk_advanced(X_realA_original)
        XA_global_splited = self.tensor_chunk_advanced(self.original_changer(x_global))
        XfakeB_splited = []
        with torch.no_grad():
            for tensor in zip(XA_original_splited, XA_global_splited):
                XfakeB_splited.append(g_model_fine(tensor[0], tensor[1]))
        XfakeB_cated = self.tensor_cat_advanced(XfakeB_splited)
        return X_fakeB_half, XfakeB_cated
    
    
if __name__=="__main__":
    original_changer = transforms.Resize((2432, 3072))
    viz = Visualizer(original_changer, '/home1/fzj/experimental_things/tb_result')
    g_model_coarse = Global_Generator(input_nc=3, output_nc=1).cuda()
    g_model_fine = Local_Enhancer(input_nc=3, output_nc=1, img_sz=256).cuda()
    F_A_dataset = slo_ffa_dataset("../dataset/data3", (2432, 3072), 256)
    val_dataloader = data.DataLoader(F_A_dataset, 1, True)
    val_iter = iter(val_dataloader)
    viz.iter_summarize_performance(g_f_model=g_model_fine, g_c_model=g_model_coarse, iter_thing=val_iter, iteration_str='1')
       
        