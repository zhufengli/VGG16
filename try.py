from PIL import Image
import cv2
import os
import numpy as np
import csv

def get_image(batch_size,dir_path):
    files = os.listdir(dir_path)
    _index = np.random.randint(0,150931,batch_size)
    print(_index)
    image_batch=np.empty([1,224,224,3])
    label_batch=np.empty([1,5])
    
    for i in range(batch_size):
        new_image=cv2.imread(os.path.join(dir_path,files[_index[i]]))
        new_image.resize(1,224,224,3)
        image_batch=np.append(image_batch,new_image,axis=0)
    image_batch=np.delete(image_batch,0,0)
    return image_batch
    

OK=get_image(10,'./train')

print (OK.shape)

