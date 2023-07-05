import argparse
from utils.functions import check_dir
from torchvision import utils
from networks.Gen_models import *
from torch import nn
import cv2 as cv
from torchvision import transforms
from utils.visualization import Pic_saver

initializer_whole = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((2432, 3072)),
    transforms.Normalize(mean=0.5, std=0.5)
])

initializer_half = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((608, 768)),
    transforms.Normalize(mean=0.5, std=0.5)
])

def main(args):
    check_dir(args.Coarse_save)
    check_dir(args.Fine_save)
    norm_layer = nn.InstanceNorm2d
    g_model_coarse = Global_Generator(input_nc=3, output_nc=1, norm_layer=norm_layer, n_downsampling=3).cuda()
    g_model_fine = Local_Enhancer(input_nc=3, output_nc=1, img_sz=256, 
                                                  norm_layer=norm_layer, n_downsampling=2).cuda()
    
    g_model_fine.load_state_dict(torch.load(args.Gen_F_path))
    g_model_coarse.load_state_dict(torch.load(args.Gen_C_path))
    original_changer = transforms.Resize((2432, 3072))
    pic_saver = Pic_saver(original_changer, Gen_C=g_model_coarse, Gen_F=g_model_fine)
    
    up_dir = args.data_path
    for i in range(4):
        slo_path = f'{args.data_path}{i+1}.png'
        ffa_path = f'{args.data_path}{i+1}-{i+1}.png'
        slo_pic = cv.cvtColor(cv.imread(slo_path), cv.COLOR_BGR2RGB)
        ffa_pic = cv.cvtColor(cv.imread(ffa_path), cv.COLOR_BGR2GRAY)
        slo_pic_whole, ffa_pic_whole = initializer_whole(slo_pic).unsqueeze(0), initializer_whole(ffa_pic).unsqueeze(0)
        slo_pic_half, ffa_pic_half = initializer_half(slo_pic).unsqueeze(0), initializer_half(ffa_pic).unsqueeze(0)
        pic_saver.save_img([slo_pic_whole, ffa_pic_whole, slo_pic_half, ffa_pic_half], index=i+1, 
                             Coarse_save=args.Coarse_save, Fine_save=args.Fine_save)
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Gen_C_path', type=str, default='checkpoints/g_model_coarse.pt')
    parser.add_argument('--Gen_F_path', type=str, default='checkpoints/g_model_fine.pt')
    parser.add_argument('--Coarse_save', type=str, default='result_save/Coarse_result/')
    parser.add_argument('--Fine_save', type=str, default='result_save/Fine_result/')
    parser.add_argument('--data_path', type=str, default='example_pics/')
    
    args = parser.parse_args()
    
    main(args)