import pandas
import os
import numpy as np

label_path = r'F:\dataset\kaggle\whale\bounding_boxes.csv'
image_path = r'F:\dataset\kaggle\whale\train'
output_file = r'train.txt'
file_list = os.listdir(image_path)

df = pandas.read_csv(label_path)

with open(output_file,'w') as f:
    for data in df.values:
        if data[0] in file_list:
            file_name = os.path.join(image_path, data[0])
            f.write(file_name+' '+','.join([str(i) for i in data[1:]])+','+str(0))
            f.write('\n')

