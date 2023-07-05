import cv2 as cv
import glob
import os 




def jpg2png(dir):
    file_name_list = glob.glob(dir+'*.jpg')
    for i in file_name_list:
        new_name =  i.split('/')[-1].split('.')[0] + '.png'
        new_path = dir + new_name
        src = cv.imread(i)
        cv.imwrite(new_path, src)
        os.remove(i)

# First postion for slo and second for ffa
def split_data(up_dir, to_dir: tuple):
    name_list = glob.glob(up_dir+'*.png')
    name_len = len(name_list)
    enum_num = name_len//2
    for i in to_dir:
        if not os.path.exists(i):
            os.makedirs(i)

    for i in range(1, enum_num+1):
        slo = cv.imread(f'{up_dir}{i}.png')
        ffa = cv.imread(f'{up_dir}{i}-{i}.png')
        cv.imwrite(f'{to_dir[0]}{i}.png', slo)
        cv.imwrite(f'{to_dir[1]}{i}-{i}.png', ffa)
        
        
        
    
    
if __name__ == "__main__":
    dir = 'dataset/registrated_sets/'
    slo_path = 'dataset/SLO_path/'
    ffa_path = 'dataset/FFA_path/'
    split_data(dir, (slo_path, ffa_path))
