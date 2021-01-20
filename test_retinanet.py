import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from matplotlib import pyplot as plt
import glob


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, class_name, dest_path, model_path, class_list):
    
    dir = os.path.join(dest_path,"vetricular_crop")
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    dir = os.path.join(dest_path,"vetricular_mitral")
    if not os.path.exists(dir):
        os.mkdir(dir)
    

    dir = os.path.join(dest_path,"selected_frames")
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir = os.path.join(dest_path,"mitral_valve_crop")
    if not os.path.exists(dir):
        os.mkdir(dir)
    

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    area_vent=[]
    #print('Detecting LV and analysing cardiac phase')
    for img_name in os.listdir(image_path):
        
        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            #print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            #print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            
            cnt=0
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                #print(bbox, classification.shape)
                score = scores[j]               
                w1=x2-x1
                h1=y2-y1
                area1=w1*h1
                caption = '{} {:.3f}'.format(label_name, score )
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                image_orig0=image_orig
                
                
                if label_name == 'vetricular':
                    crop_img = image_orig0[y1:y2, x1:x2,:]
                    gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray_image,(5,5),0)
                    ret3,th3 = cv2.threshold(blur,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    TotalNumberOfPixels = crop_img.shape[0] * crop_img.shape[1]
                    non_zero=cv2.countNonZero(th3)
                    ZeroPixels = TotalNumberOfPixels - cv2.countNonZero(th3)
                    
                    #cv2.imshow('cropped', th3)
                    #cv2.waitKey(0)
                    area_vent.append(ZeroPixels)
                    ##########cv2.imwrite(dest_path +  "/vetricular_crop/" + class_name + '_' +  os.path.basename(image_path) + '_' + img_name  , crop_img)

                #draw_caption(image_orig, (x1, y1, x2, y2), caption + ' ' + str(area1))
                #cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)


            #cv2.imshow('detections', image_orig)
            #cv2.waitKey(0)
            #######################cv2.imwrite(dest_path +  "/vetricular_mitral/" +  img_name , image_orig)
    


    len_cycle=(len(area_vent)) 
    m_area_vent=(max(area_vent))
    area_vent_norm = [x / m_area_vent for x in area_vent]
    min_index=np.argmin(area_vent)
    x_axes=list(range(0, len_cycle))
    #plt.plot(area_vent_norm)
    #fig = plt.figure()
    #ax = plt.subplot(111)
    #ax.plot(x_axes, area_vent_norm)
    #plt.title('Left Ventrucular volume')
    #plt.xlabel('Number of frames')
    #plt.ylabel('normalized volume')
    ###############fig.savefig(plot_path + 'Plot_' + class_name + '_' +  os.path.basename(image_path) + '_' + img_name )
    #plt.plot(area_vent_norm)
    #plt.xlabel('Number of frames')
    #plt.ylabel('normalized volume')
    #plt.show()
    #####################plt.savefig(image_path +  "/vetricular_crop/"  + 'volume.jpg')

    #print(min_index)
    #selected_frames=[min_index : len_cycle-1]
  
    
    im_list=glob.glob(image_path + '/' + "*.jpg")

    selected_image=im_list[min_index : len_cycle-1]
    #print('Selecting Diastolic frames')
    for S_img_name in selected_image:
        
        path, filename = os.path.split(S_img_name)
        #print(' Path is %s and file is %s' % ( path, filename))
        #print(os.path.basename(path))
        image = cv2.imread(S_img_name)

        #print(filename)
        ##cv2.imwrite(dest_path +  "/selected_frames/" + os.path.basename(path) + '_' + filename , image)
        #####cv2.imwrite(dest_path +  "/selected_frames/" +  class_name + '_' + os.path.basename(path) + '_' + filename , image)
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            #print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            #print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            
            cnt=0
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                #print(bbox, classification.shape)
                score = scores[j]               
                w1=x2-x1
                h1=y2-y1
                area1=w1*h1
                caption = '{} {:.3f}'.format(label_name, score )
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                image_orig0=image_orig
                
                
                if label_name == 'vetricular':
                    crop_img = image_orig0[y1:y2, x1:x2,:]
                    ###cv2.imwrite(image_path +  "/mitral_valve_crop/" + os.path.basename(path) + '_' + filename , crop_img)
                    cv2.imwrite(dest_path+  "/vetricular_crop/" + class_name + '_' +  os.path.basename(path) + '_' + filename , crop_img)
                #draw_caption(image_orig, (x1, y1, x2, y2), caption + ' ' + str(area1))
                #cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', default = './data/' , help='Path to directory containing images')
    parser.add_argument('--model_path', default = './Object_Detection/pytorch-retinanet-master/train_models/A4C/resnet152/model_final_152_m95_v99.pt' , help='Path to model')
    parser.add_argument('--class_list', default = './2class.csv' , help='Path to CSV file listing class names (see README)')
    parser.add_argument('--dest_path', default = './dest_path/' , help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()
    root = parser.image_dir
    file_err = open(parser.dest_path + '/error.txt',"w")

    class_name=[ "III" ]
    data_name=[  "train" ,"valid"]

    plot_path= parser.dest_path +'plots/' 

    for dn in data_name:
        print(dn)
        for cn in class_name:

            print(cn)

            parent_path_root=root + '/'  + dn + "/Selected_" + dn + '_' +  cn + "/A4C"#Source folder

            dest_file=parser.dest_path + '/'  + dn + '/' +  cn + "/"#Destination folder
        
        
            for cases in os.listdir(parent_path_root):
                try:
                    print(cases)
                    detect_image(parent_path_root + '/' + cases, cn, dest_file, parser.model_path, parser.class_list)
                except:
                    file_err.write(cn + '  ' + cases + '\n' )
    file_err.close()