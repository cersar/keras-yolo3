import cv2
from PIL import Image
from yolo import YOLO
import numpy as np
import os
import pandas

label_path = r'F:\dataset\kaggle\whale\bounding_boxes.csv'
image_path = r'F:\dataset\kaggle\whale\train'
file_list = os.listdir(image_path)
obj = YOLO()
for file in file_list:
    filename = os.path.join(image_path, file)
    img = Image.open(filename)
    result = obj.detect_image(img)
    cv2.imshow(file, np.asarray(result))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# df = pandas.read_csv(label_path)

# for data in df.values:
#     filename = os.path.join(image_path,data[0])
#     if os.path.exists(filename):
#         img = cv2.imread(filename)
#         cv2.rectangle(img,tuple(data[1:3]),tuple(data[3:]),(0,0,255))
#         cv2.imshow(data[0],img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# obj.detect_image()