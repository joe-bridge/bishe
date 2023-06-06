from lowlight_enhancement import lowlight_enhancement
import cv2
import os
file_path = "data/MOT20-02/img1"
img_file_list = os.listdir(file_path)
for img_file in img_file_list:
    image_path = os.path.join(file_path, img_file)
    image = cv2.imread(image_path)
    image = lowlight_enhancement(image)
    cv2.imwrite(os.path.join(file_path, img_file), image)
