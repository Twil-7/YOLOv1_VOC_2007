import os
import xml.etree.ElementTree as ET


class_dictionary = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
                    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
                    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
                    'sofa': 17, 'train': 18, 'tvmonitor': 19}
class_list = list(class_dictionary.keys())


def read_image_path():
    data_x = []
    filename = os.listdir('JPEGImages')
    filename.sort()
    for name in filename:
        path = 'JPEGImages/' + name
        data_x.append(path)

    print('JPEGImages has been download ! ')
    return data_x


def read_coordinate_txt():
    data_y = []
    filename = os.listdir('Annotations')
    filename.sort()
    for name in filename:
        tree = ET.parse('Annotations/' + name)
        root = tree.getroot()

        coordinate = ''
        for obj in root.iter('object'):

            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in class_list or int(difficult) == 1:
                continue

            cls_id = class_list.index(cls)
            xml_box = obj.find('bndbox')
            x_min = int(xml_box.find('xmin').text)
            y_min = int(xml_box.find('ymin').text)
            x_max = int(xml_box.find('xmax').text)
            y_max = int(xml_box.find('ymax').text)

            loc = (str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + ',' + str(cls_id) + '  ')
            coordinate = coordinate + loc

        data_y.append(coordinate)

    print('Object Coordinate has been download ! ')
    return data_y


def make_data():
    data_x = read_image_path()
    data_y = read_coordinate_txt()

    n = len(data_x)
    train_x = data_x[0:9000]
    train_y = data_y[0:9000]
    val_x = data_x[9000:9500]
    val_y = data_y[9000:9500]
    test_x = data_x[9500:n]
    test_y = data_y[9500:n]

    return train_x, train_y, val_x, val_y, test_x, test_y


