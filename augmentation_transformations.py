"""
example transformatin:
[
    A.RandomCrop(width=640, height=640),
    A.Rotate(border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5),
    ], p=1.0),
    A.RandomBrightnessContrast()
]

See more transformations here: https://albumentations.ai/docs/api_reference/full_reference/
"""
import albumentations as A
import cv2

transformations_dict = {
    0: [

    ],
    1: [
        A.HorizontalFlip(),
        A.VerticalFlip()
    ],
    2: [
        A.Rotate(border_mode=cv2.BORDER_CONSTANT)
    ],
    3: [
        A.ColorJitter()
    ],
    4: [
        A.RandomBrightnessContrast()
    ],
    5: [
        A.Blur(blur_limit=3)
    ],
    6: [
        A.Blur(blur_limit=2)
    ],
    7: [
        A.Blur(blur_limit=1)
    ],
    8: [
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25)
    ],
    9: [
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15)
    ],
    10: [
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
    ],
    11: [
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5)
    ]
}