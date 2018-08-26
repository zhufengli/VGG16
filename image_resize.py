from PIL import Image
import os

def resize_image(dir_path):
    dirs = os.listdir(dir_path)
    counter=0
    for dir in dirs:
        son_dir_path = os.path.join(dir_path+'/'+dir)
        files = os.listdir(son_dir_path)
        for file in files:
            name=os.path.join(son_dir_path,file)
            with Image.open(name) as raw_im:
                if raw_im.size[0]>100 and raw_im.size[1]>100:
                    im=raw_im.resize((224,224))
                    im.save(name,'JPEG')
                else: 
                    os.remove(name)
            counter = counter + 1
    print("Number image processed : ", counter)

resize_image('./train')
