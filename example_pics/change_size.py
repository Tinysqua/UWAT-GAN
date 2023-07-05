import cv2 as cv
from glob import glob
import os

address_list = glob('*.png')
print(len(address_list))
for i in address_list:
    pic = cv.imread(i)
    pic = cv.resize(pic, dsize=(3072, 2432))
    os.remove(i)
    cv.imwrite(i, pic)