import os

from PIL import Image

file_path = './'
file_list = os.listdir(file_path)
for file in file_list:
    img = Image.open(os.path.join(file_path, file))
    dark_img = Image.blend(img, Image.new('RGB', img.size, (0, 0, 0)), 0.5)
    dark_img.save(os.path.join(file_path, file))