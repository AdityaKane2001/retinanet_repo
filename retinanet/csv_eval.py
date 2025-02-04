from __future__ import print_function

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
import cv2
import pandas as pd
from google.colab import data_table


def get_annots_from_txt(label_file_path):
    try:
        file_handler = open(label_file_path,'r')
    except :
        return np.array([],dtype = np.float32),np.array([],dtype = np.float32),np.array([],dtype = np.float32)
    print(label_file_path)
    lines = file_handler.readlines()
    read_lines=[]
    for i in lines:
        line = i.strip('\n')
        num_list = line.split(' ')
        read_lines.append(list(map(lambda x:float(x),num_list)))
    file_handler.close()
    
    df = pd.DataFrame(read_lines)
    
    if df.empty:
        return np.array([],dtype = np.float32),np.array([],dtype = np.float32),np.array([],dtype = np.float32)
    scores = np.array(df[5],dtype = np.float32)
    labels = np.array(df[0],dtype = np.int64)
    x1 = list(df[1])
    y1 =  list(df[2])
    x2 =  list(df[3])
    y2 = list(df[4])
    boxes=[]
    for j in range(len(labels)):
        boxes.append([x1[j],y1[j],x2[j],y2[j]])
    
    boxes = np.array(boxes,dtype = np.float32)
    
    return scores, labels, boxes

def _get_detections_from_txt(dataset, labels_dir, score_threshold=0.001, max_detections=100, save_path=None):
    
    #dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
    #                           transform=transforms.Compose([Normalizer(), Resizer()]))
    #return 0
    
    labels_list = os.listdir(labels_dir)


    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            txt_path =os.path.join(labels_dir, data['image_path'].split('/')[-1][:-4]+'.txt') 
            scores,labels,boxes = get_annots_from_txt(txt_path)
            img = cv2.imread(data['image_path'])
            # run network
            # if torch.cuda.is_available():
            #     scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            # else:
            #     scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            # scores = scores.cpu().numpy()
            # labels = labels.cpu().numpy()
            # boxes  = boxes.cpu().numpy()
            
            # correct boxes for image scale
            boxes /= 416/640
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections

    



def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():
        if save_path!=None:
            save_dict=dict()
            save_dict['image_name'] = []
            save_dict['x1'] = []
            save_dict['x2'] = []
            save_dict['y1'] = []
            save_dict['y2'] = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                if save_path!=None:
                    for i in range(len(image_boxes)):
                        save_dict['image_name'].append(data['image_path'])
                        save_dict['x1'].append(image_boxes[i,0])
                        save_dict['x2'].append(image_boxes[i,1])
                        save_dict['y1'].append(image_boxes[i,2])
                        save_dict['y2'].append(image_boxes[i,3])

                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')
    if save_path!=None:
        df = pd.DataFrame(save_dict)
        df.to_csv(save_path)
    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    retinanet=None,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
    df_save_path=None,
    labels_dir=None, #provide if mode='yolo'
    mode = 'retina'
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations
    if mode=='retina':
        all_detections     = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=df_save_path)
    else:
        all_detections = _get_detections_from_txt(generator, labels_dir, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        
        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations


    print('\nmAP:')
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
        print("Precision: ",precision[-1])
        print("Recall: ",recall[-1])
        
        if save_path!=None:
            plt.plot(recall,precision)
            # naming the x axis 
            plt.xlabel('Recall') 
            # naming the y axis 
            plt.ylabel('Precision') 

            # giving a title to my graph 
            plt.title('Precision Recall curve') 

            # function to show the plot
            plt.savefig(os.path.join(save_path,label_name+'_precision_recall.jpg'))



    return average_precisions

