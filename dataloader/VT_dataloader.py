from torch.utils import data
from torchvision import transforms
from PIL import Image
import glob
from os.path import join
import albumentations as A 
import cv2 as cv
import numpy as np

def get_address_list(up_dir, picture_form):
    return glob.glob(up_dir+'*.'+picture_form)



class slo_ffa_dataset(data.Dataset):
    def __init__(self, up_dir, original_img_size, img_size):
        super(slo_ffa_dataset, self).__init__()
        fu_path = join(up_dir, "Images/")
        self.an_path = join(up_dir, "Masks/")
        self.fu_path =  get_address_list(fu_path, "png")
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        # Now we define the transforms in the dataset
        
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((original_img_size[0], original_img_size[1])),
            transforms.Normalize(mean=0.5, std=0.5)])
        
        self.transformer_mini = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Normalize(mean=0.5, std=0.5)])    
        
    def __getitem__(self, index):
        fun_filename = self.fu_path[index]
        middle_filename = fun_filename.split("/")[-1].split(".")[0]
        first_num, second_num = int(middle_filename.split("_")[0]), int(middle_filename.split("_")[1])
        
        XReal_A, XReal_A_half = self.convert_to_resize(self.funloader(fun_filename))
        an_filename = str(first_num)+"_mask_"+str(second_num)+".png"
        an_file_path = self.an_path + an_filename
        XReal_B, XReal_B_half = self.convert_to_resize(self.angloader(an_file_path))
        return [XReal_A, XReal_B, XReal_A_half, XReal_B_half]
    
    
    def convert_to_resize(self, X):
        y1 = self.transformer(X)
        y2 = self.transformer_mini(X)
        return y1, y2
    
    def __len__(self):
        return len(self.fu_path)
    
    def funloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def angloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
        


class slo_ffa_dataset_augmentation(data.Dataset):
    def __init__(self, up_dir, img_size):
        super(slo_ffa_dataset, self).__init__()
        fu_path = join(up_dir, "Images/")
        self.an_path = join(up_dir, "Masks/")
        self.fu_path =  get_address_list(fu_path, "png")
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        # Now we define the transforms in the dataset
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Normalize(mean=0.5, std=0.5)])
        
        self.transformer_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0]//2, img_size[1]//2)),
            transforms.Normalize(mean=0.5, std=0.5)])    
        
        self.augmentation = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(), 
            A.ShiftScaleRotate(shift_limit_y=0.02, rotate_limit=40)
        ])
        
    def __getitem__(self, index):
        fun_filename = self.fu_path[index]
        middle_filename = fun_filename.split("/")[-1].split(".")[0]
        first_num, second_num = int(middle_filename.split("_")[0]), int(middle_filename.split("_")[1])
        A_numpy = self.funloader(fun_filename)
        
        an_filename = str(first_num)+"_mask_"+str(second_num)+".png"
        an_file_path = self.an_path + an_filename
        B_numpy = np.expand_dims(self.angloader(an_file_path), axis=2)
        
        AB_cat = np.concatenate((A_numpy, B_numpy), axis=2)
        augmented_AB = self.augmentation(image=AB_cat)['image']
        XReal_A, XReal_A_half = self.convert_to_resize(augmented_AB[:,:,:3])
        XReal_B, XReal_B_half = self.convert_to_resize(augmented_AB[:,:,3])
        return [XReal_A, XReal_B, XReal_A_half, XReal_B_half]
    
    
    def convert_to_resize(self, X):
        y1 = self.transformer(X)
        y2 = self.transformer_resize(X)
        return y1, y2
    
    def __len__(self):
        return len(self.fu_path)
    
    def funloader(self, path):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
        
    def angloader(self, path):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img
           