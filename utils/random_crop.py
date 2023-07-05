from PIL import Image  # PIL = Python Image Library
import numpy as np
import os  # 在python下写程序，需要对文件以及文件夹或者其他的进行一系列的操作，os便是对文件或文件夹操作的一个工具。
import random
import argparse  # argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
import glob

# dir means directoty
def random_crop(img, mask, width, height, num_of_crops,name,stride=1,dir_name='data'):
    Image_dir = dir_name + 'Images'  # data/Images
    Mask_dir = dir_name + 'Masks'   # data/Masks
    directories = [dir_name,Image_dir,Mask_dir]  # 把它们放入一个数组中
    # 检查文件要写入的目录是否存在？
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    max_y = int(((img.shape[0]-height)/stride)+1)
    max_x = int(((img.shape[1]-width)/stride)+1)

    # 输出0-max_x-1
    crop_x = [i for i in range(0, max_x)]
    # 输出0-max_y-1
    crop_y = [i for i in range(0, max_y)]
    for i in range(1,num_of_crops+1):
        x_seq = random.choice(crop_x)
        y_seq = random.choice(crop_y)

        crop_img_arr = img[y_seq:y_seq+height, x_seq:x_seq+width]

        crop_mask_arr = mask[y_seq:y_seq+height, x_seq:x_seq+width]
        crop_img = Image.fromarray(crop_img_arr)
        crop_mask = Image.fromarray(crop_mask_arr)
        img_name = directories[1] + "/" + name + "_" + str(i)+".png"
        mask_name = directories[2] + "/" + name + "_mask_" + str(i)+".png"
        crop_img.save(img_name)
        crop_mask.save(mask_name)




def resize_img(image_png, mask_png,height, width,dir_name = 'data_resize'):
    All_Image_dir = dir_name + '/All_Images'
    directories = [dir_name, All_Image_dir]
    # 检查文件要写入的目录是否存在？
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


    img = image_png.resize((height, width))
    mask = mask_png.resize((height, width))
    return img, mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # history: (1451, 1109), (2600, 2048)
    parser.add_argument('--width', type=int, default=3840)
    parser.add_argument('--height', type=int, default=3040)
    parser.add_argument('--datadir', type=str, help='path/to/data_directory', default='dataset/registrated_sets/')
    
    # history: (1088, 832)
    parser.add_argument('--input_dim_width', type=int, default=3072)
    parser.add_argument('--input_dim_height', type=int, default=2432)
    parser.add_argument('--n_crops', type=int, default=40)
    # parser.add_argument('--output_dir', type=str, default='dataset/data/')
    parser.add_argument('--output_dir', type=str, default='dataset/data3/')
    args = parser.parse_args()


    # for i in range(1, 4):
    #     img_name = args.datadir + "/" + str(i) + ".png"
    #     img_png = Image.open(img_name)
    #     mask_name = args.datadir + "/" + str(i) + "-" + str(i) + ".png"
    #     mask_png = Image.open(mask_name)
    #     name = str(i)
    #     resize_img(img_png, mask_png, args.height, args.width)
    file_name_list = glob.glob(args.datadir + '*.png')
    len_file = len(file_name_list)
    updir = args.datadir
    new_up_dir = args.output_dir
    size = (args.width, args.height)
    bias = 0
    
    for i in range(bias+1, bias+len_file//2+1):
        old_slo_path = updir + str(i) + '.png'
        old_ffa_path = updir + str(i) + '-' + str(i) + '.png'
        old_slo = Image.open(old_slo_path)
        old_ffa = Image.open(old_ffa_path)
        new_slo = np.asarray(old_slo.resize(size))
        new_ffa = np.asarray(old_ffa.resize(size))
        name = str(i)
        random_crop(new_slo, new_ffa, args.input_dim_width, args.input_dim_height, args.n_crops, name, dir_name=args.output_dir)
        
        


    # for i in range(1, 4):
    #     img_name = args.datadir + "/" + str(i) + "_resize" + ".png"
    #     im = Image.open(img_name)
    #     img_arr = np.asarray(im)
    #     mask_name = args.datadir + "/" + str(i)+"-" + str(i) + "_resize" + ".png"
    #     mask = Image.open(mask_name)
    #     mask_arr = np.asarray(mask)
    #     name = str(i)
    #     random_crop(img_arr, mask_arr, args.input_dim_1, args.input_dim_2, args.n_crops, name)