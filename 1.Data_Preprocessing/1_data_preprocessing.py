# import modules
import labelme
import os,sys
import shutil
import warnings

# ignore warnings
warnings.filterwarnings('ignore')

# getting current working directory
os.getcwd()

# delete existing folders
def delete_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
delete_folder(os.path.join(os.getcwd(),'temp_images/'))
delete_folder(os.path.join(os.getcwd(),'masked_images/'))
delete_folder(os.path.join(os.getcwd(),'original_images/'))

# make directories
os.makedirs('temp_images')
os.makedirs('masked_images')
os.makedirs('original_images')


# generating mask images
def generate_mask(path):
    dirs = os.listdir(path)
    i = 0
    for item in dirs:
        if item.endswith(".json"):
            if os.path.isfile(path + item):
                my_dest = "temp_images/" + item[:-5]
                os.system("mkdir " + my_dest)
                print("image:", item)
                os.system("labelme_export_json " + path + item + " -o " + my_dest)
                i = i + 1

path=os.path.join(os.getcwd(),'AnnotatedImages/')
generate_mask(path)

# delete empty folders
def delete_empty_folders(root):
    deleted = set()

    for current_dir, subdirs, files in os.walk(root, topdown=False):

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break

        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)


def delete_empty_folders(root):
    deleted = set()

    for current_dir, subdirs, files in os.walk(root, topdown=False):

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break

        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)

path=os.path.join(os.getcwd(),'temp_images/')
delete_empty_folders(path)

# separating original and masked images
def mask_images(dir):
    new_masked_image = []
    original_image=[]
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name=='label.png':
                shutil.move(root+'/'+name,masked_path+str(root).split('/')[-1]+'.png')
                new_masked_image.append(os.path.join(root, name))
            elif name=='img.png':
                shutil.move(root+'/'+name,original_path+str(root).split('/')[-1]+'.png')
                original_image.append(os.path.join(root, name))
masked_path=os.path.join(os.getcwd(),'masked_images/')
original_path=os.path.join(os.getcwd(),'original_images/')
mask_images(path)
print("Stage 1 is finished")