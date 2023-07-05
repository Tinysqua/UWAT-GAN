from torch.utils import data
from torchvision import transforms
from PIL import Image
import glob
from os.path import join

def get_address_list(up_dir, picture_form: str):
    return glob.glob(up_dir+'*.'+picture_form)

def int_2_tuple(size):
    if isinstance(size, int):
        return(size, size)
    else:
        return size

class Evaluation_dataset(data.Dataset):
    def __init__(self, up_dir, original_img_size, img_size):
        super(Evaluation_dataset, self).__init__()
        slo_path = join(up_dir, 'SLO_path/')
        self.ffa_up = join(up_dir, 'FFA_path/')
        self.slo_path = get_address_list(slo_path, 'png')
        img_size = int_2_tuple(img_size)
        original_img_size = int_2_tuple(original_img_size)
        # if isinstance(img_size, int):
        #     self.img_size = (img_size, img_size)
        
            
    # Now we define the transforms in the dataset
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((original_img_size[0], original_img_size[1])),
            transforms.Normalize(mean=0.5, std=0.5)])

        self.transformer_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Normalize(mean=0.5, std=0.5)])  
        
    def __getitem__(self, index):
        slo_file_name = self.slo_path[index]
        middle_filename = slo_file_name.split("/")[-1].split(".")[0]
        ffa_file_name = f'{self.ffa_up}{middle_filename}-{middle_filename}.png'
        XReal_A, XReal_A_half = self.convert_to_resize(self.funloader(slo_file_name))
        XReal_B, XReal_B_half = self.convert_to_resize(self.angloader(ffa_file_name))
        return [XReal_A, XReal_B, XReal_A_half, XReal_B_half]

        
    def convert_to_resize(self, X):
        y1 = self.transformer(X)
        y2 = self.transformer_resize(X)
        return y1, y2
        
    def __len__(self):
        return len(self.slo_path)
    
    def funloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def angloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
