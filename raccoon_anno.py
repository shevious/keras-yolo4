import xml.etree.ElementTree as ET
from os import getcwd

#sets=[('2012', 'train'), ('2012', 'val'), ('2012', 'test')]
sets=[('raccoon', 'train'), ('raccoon', 'test')]

classes = ["raccoon"]


def convert_annotation(year, image_id, list_file):
    in_file = open('data/raccoon_dataset/annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for year, image_set in sets:
    in_file = open('data/raccoon_dataset/data/%s_labels.csv'%(image_set))
    lines = in_file.readlines()
    image_ids = []
    for line in lines:
      image_ids.append(line.split('.')[0])
    image_ids = image_ids[1:]
    #print(image_ids)
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/data/raccoon_dataset/images/%s.jpg'%(wd, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()

