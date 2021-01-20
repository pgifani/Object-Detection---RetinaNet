import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

im_path='./train/JpegImages/'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
.
            value = (im_path + root.find('filename').text,
                     
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     member[0].text,
                     )
            xml_list.append(value)

    column_name = ['filename',   'xmin', 'ymin', 'xmax', 'ymax' ,'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    
    return xml_df


def main():
    image_path = os.path.join('./valid/', 'Annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('./train/A4C_valid.csv', index=None)
    print('Successfully converted xml to csv.')


main()