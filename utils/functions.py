import yaml
import os

def convert_to_cuda(x, device=None):
    if device is None:
        return x.cuda()
    else:
        return x.to(device)


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config


def check_dir(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)
    return dire
        
def compute_max_xy(large_img_size, patch_img_size):
    max_y = int((large_img_size[0]-patch_img_size)+1)
    max_x = int((large_img_size[1]-patch_img_size)+1)
    return max_y, max_x

