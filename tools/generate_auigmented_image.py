import albumentations as A
import cv2
import random

img_path = "data/content-images/berlin_000000_000019_leftImg8bit.png"

random.seed(0)
transform = A.Compose(
    [A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)],
)

image = cv2.imread(img_path)
aug_image = transform(image=image)["image"]
cv2.imwrite("test.jpg", aug_image)
