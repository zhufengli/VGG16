import skimage
import skimage.io
import skimage.transform
import os

def resize_image(dir_path):
    dirs = os.listdir(dir_path)
    counter=0
    for dir in dirs:
        son_dir_path = os.path.join(dir_path+'/'+dir)
        files = os.listdir(son_dir_path)
        for file in files:
            name=os.path.join(son_dir_path,file)
            img = skimage.io.imread(name)
            if img.shape[0]>100 and img.shape[1]>100:
                short_edge = min(img.shape[:2])
                yy = int((img.shape[0] - short_edge) / 2)
                xx = int((img.shape[1] - short_edge) / 2)
                crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
                resized_img = skimage.transform.resize(crop_img, (224, 224))
                skimage.io.imsave(name,resized_img)
            else: 
                os.remove(name)
            counter = counter + 1
    print("Number image processed : ", counter)

resize_image('./train')
