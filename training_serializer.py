"""
Trains the model multiple times with different augmentation parameters

**Place this in the yolo folder**
"""

import os
from cmath import inf
import cv2
import albumentations as A
import numpy as np
from PIL import Image
import os, os.path
from datetime import datetime
import torch

def main():
    # define augmentation transformations
    augmentation_transformations = [
            [
                # A.RandomCrop(width=640, height=640),
                A.HorizontalFlip(),
                # A.crop_coords
            ],
            [
                # A.RandomCrop(width=640, height=640),
                A.VerticalFlip(),
                # A.crop
            ],       
    ]


    # training parameters
    PROJECT_NAME = "Merged-sheep-dataset"
    PROJECT_VERSION = 3
    PROJECT_DIRECTORY = f"{PROJECT_NAME}-{PROJECT_VERSION}"

    AUGMENTATION = True
    # AUGMENTATIONS = float('inf')
    AUGMENTATIONS = 100
    EPOCHS = 1

    # remove potential residual augmentations
    # remove_augmentations(True) # todo find all augmentation images

    # add augmentations

    def create_augmentations(AUGMENTATIONS):
        # reset labels cache
        os.system("rm Merged-sheep-dataset-3/train/labels.cache -f")
        CLASS_LABEL = 'sheep'
        CLASS = 0

        # load train images
        images = []
        images_load_path = f"./{PROJECT_DIRECTORY}/train/images/"
        valid_images = [".jpg",".gif",".png",".tga"]

        LIMIT = AUGMENTATIONS
        i = 0
        for image_file_name in os.listdir(images_load_path):
            if i == LIMIT:
                break
            ext = os.path.splitext(image_file_name)[1]
            if ext.lower() not in valid_images:
                continue
            images.append(cv2.imread(os.path.join(images_load_path, image_file_name)))
            i += 1


        # load train labels
        labels = [] # one label per image, multiple bboxes per image
        labels_load_path = f"./{PROJECT_DIRECTORY}/train/labels/"
        i = 0
        for label_file_name in os.listdir(labels_load_path):
            if i == LIMIT:
                break
            label = open(os.path.join(labels_load_path, label_file_name), "r")
            bboxes = []
            for bbox in label:
                bbox = np.array(bbox.split(' ')[1:]).astype(np.float32)
                bboxes.append(bbox)
            labels.append(bboxes)
            i += 1



        # augment images using transformation

        augmented_labels = []
        augmented_images_nd = []
        for image, label in zip(images, labels):
            label = np.array(label)

            transform = A.Compose(
                augmentation_transformation,
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
            )
            
            augmentation = transform(image=image, bboxes=label, class_labels=[CLASS_LABEL]*len(label))
            augmented_img = augmentation["image"]
            augmented_label = augmentation["bboxes"]
            augmented_labels.append(augmented_label)
            augmented_images_nd.append(augmented_img)
            


        augmented_images = [Image.fromarray(augmented_image_nd) for augmented_image_nd in augmented_images_nd]

        # create image save dir
        img_save_dir = f"./{PROJECT_DIRECTORY}/train/images"


        # save augmented images
        for i in range(len(augmented_images)):
            image_file_name = f"{i}.png"
            image_dir = os.path.join(img_save_dir, image_file_name)
            augmented_images[i].save(image_dir)
            print(f"saved augmented image: {image_dir}")


        # create labels save dir
        labels_save_dir = f"./{PROJECT_DIRECTORY}/train/labels"

        # write augmented labels
        for i in range(len(augmented_labels)):
            label_file_name = f"{i}.txt"
            label_dir = os.path.join(labels_save_dir, label_file_name)
            with open(label_dir, 'w') as f:
                for bbox in augmented_labels[i]:
                    f.write(f"{CLASS} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            print(f"saved augmented label: {label_dir}")


    def remove_augmentations(AUGMENTATIONS, all=False):
        """
        all: removes all image augmentations
        """
        if all:
            AUGMENTATIONS = float('inf')

        # remove augmented images
        img_save_dir = f"./{PROJECT_DIRECTORY}/train/images"
        for i in range(int(AUGMENTATIONS)):
            image_file_name = f"{i}.png"
            image_dir = os.path.join(img_save_dir, image_file_name)
            os.remove(os.path.join(img_save_dir, image_file_name))
            print(f"removed image: {image_dir}")

        # remove augmented labels
        labels_save_dir = f"./{PROJECT_DIRECTORY}/train/labels"
        for i in range(int(AUGMENTATIONS)):
            label_file_name = f"{i}.txt"
            label_dir = os.path.join(labels_save_dir, label_file_name)
            os.remove(os.path.join(labels_save_dir, label_file_name))
            print(f"removed label: {label_dir}")

    # loop through training phases with different augmentations
    for i, augmentation_transformation in enumerate(augmentation_transformations):
        start_timestamp = datetime.now()
        start_timestamp = str(start_timestamp).replace(' ', '_')

        if AUGMENTATION: 
            create_augmentations(AUGMENTATIONS)

        # image results from previous runs are deleted
        os.system("rm runs/train/* -rf ")
        os.system("rm runs/detect/* -rf")

        # gpu memory is cleared
        torch.cuda.empty_cache()

        # train model
        os.system(f"python3 train.py --batch-size 48 --epochs {EPOCHS} --data Merged-sheep-dataset-3/data.yaml --weights 'yolov7-tiny.pt' --device 0 --img 640 640")

        # remove augmentations
        # todo check if dir not empty

        if AUGMENTATION: 
            remove_augmentations(AUGMENTATIONS)

        # detect
        os.system("python3 detect.py --weights runs/train/exp/weights/best.pt --conf 0.1 --source Merged-sheep-dataset-3/train/images")

        # export    
        report = f"""augmentations: {augmentation_transformation}
        dataset: {PROJECT_DIRECTORY}
        augmentations: {AUGMENTATIONS}
        epochs: {EPOCHS}
        """

        text_file = open("report.txt", "w")
        n = text_file.write(report)
        text_file.close()

        os.system(f"zip -r export_{start_timestamp}.zip report.txt")
        os.system(f"zip -r export_{start_timestamp}.zip runs/detect")
        os.system(f"zip -r export_{start_timestamp}.zip runs/train/exp/weights/best.pt")
        os.system(f"zip export_{start_timestamp}.zip runs/train/exp/*")

if __name__ == "__main__":
    main()