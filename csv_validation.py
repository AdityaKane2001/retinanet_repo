import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--mode',default='retina',help="yolo/retinanet",type=str)
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser.add_argument('--PR_save_path',help='Path to store PR curve image',default=None)
    parser.add_argument('--df_save_path')
    parser.add_argument('--yolo_labels_dir',default='',type=str,help='Path to labels_dir folder for YOLO detections')
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    if parser.mode=='retina':
        retinanet=torch.load(parser.model_path)

        use_gpu = True

        if use_gpu:
            if torch.cuda.is_available():
                retinanet = retinanet.cuda()

        if torch.cuda.is_available():
            #retinanet.load_state_dict(torch.load(parser.model_path))
            retinanet = torch.nn.DataParallel(retinanet).cuda()
        else:
            retinanet.load_state_dict(torch.load(parser.model_path))
            retinanet = torch.nn.DataParallel(retinanet)

        retinanet.training = False
        retinanet.eval()
        retinanet.module.freeze_bn()

        print(csv_eval.evaluate(dataset_val,df_save_path=parser.df_save_path,retinanet = retinanet,iou_threshold=float(parser.iou_threshold),save_path=parser.PR_save_path))
    else:
        print(csv_eval.evaluate(dataset_val,mode='yolo', labels_dir = parser.yolo_labels_dir,iou_threshold=float(parser.iou_threshold),save_path=parser.PR_save_path))



if __name__ == '__main__':
    main()
