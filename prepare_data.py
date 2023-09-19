import os
import numpy as np
import random
import json
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


data_root = './data/train'
test_root = './data/test'
processed_root = './data/processed'

with open('camera_ground_truth.json', "r") as f:
    target = json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_num', type=int, default=32, help="crop numbers for each image")
    parser.add_argument('--crop_size', type=int, default=224, help="input image size for network")
    parser.add_argument('--crop_test_dataset', help="whether to crop test folder in advance", default=False, action='store_true')
    args = parser.parse_args()
    return args

def crop_image(image_path, camera_name, crop_num, crop_size, save_path):
    data = []
    img = np.asarray(Image.open(image_path))
    h, w, _ = img.shape
    i = 0
    while i < crop_num:
        hstart = random.randint(0, h-crop_size)
        wstart = random.randint(0, w-crop_size)
        img_crop = img[hstart:hstart+crop_size, wstart:wstart+crop_size, :]

        crop_name = image_path.split('/')[-2] + '_' + image_path.split('/')[-1][:-4] + '_' + str(i) + '.jpg'
        file_path = os.path.join(processed_root, save_path, crop_name)
        
        img_crop = Image.fromarray(img_crop)
        img_crop.save(file_path, quality=100, subsampling=0)

        line = ("%s %s\n" % (file_path, camera_name))
        data.append(line)
        i += 1
    
    with open(os.path.join(processed_root, save_path + '.txt'), "a") as f:
        f.writelines(data)



def main(args):

    os.makedirs(os.path.join(processed_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(processed_root, 'val'), exist_ok=True)

    try:
        os.remove(os.path.join(processed_root, 'train.txt'))
        os.remove(os.path.join(processed_root, 'val.txt'))
    except:
        print('No old label files exist. Process to the next step...')

    for file in os.listdir(data_root):
        # Folder path for each camera device
        current_pth = os.path.join(data_root, file)
        if os.path.isdir(current_pth):
            all_images = os.listdir(current_pth)

            for idx in range(len(all_images)):
                image_path = os.path.join(current_pth, all_images[idx])
                camera_name = file.split('_', 1)[1]
                # labels = target[camera_name]

                # Create cropped image sets for training and validation
                # Keep 9 images in each folder for validation
                if idx >= len(all_images) - 9:
                    crop_image(image_path, camera_name, args.crop_num, args.crop_size, 'val')
                else:
                    crop_image(image_path, camera_name, args.crop_num, args.crop_size, 'train')

            print("Cropped for: {} in training dataset".format(file))


    ##### Create crops for test dataset if specified #####
    if args.crop_test_dataset:
        print("---- Start to crop images for test dataset")
        os.makedirs(os.path.join(processed_root, 'test'), exist_ok=True)
        try:
            os.remove(os.path.join(processed_root, 'test.txt'))
        except:
            print('No test label file found. Continue...')
    
        for file in os.listdir(test_root):
            current_pth = os.path.join(test_root, file)
            if os.path.isdir(current_pth):

                all_images = os.listdir(current_pth)
                for idx in range(len(all_images)):
                    image_path = os.path.join(current_pth, all_images[idx])
                    camera_name = file.split('_', 1)[1]
                    crop_image(image_path, camera_name, args.crop_num, args.crop_size, 'test')


if __name__ == '__main__':
	args = parse_args()
	main(args)