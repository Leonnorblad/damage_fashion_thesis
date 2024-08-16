#%%
import torch
import torchvision
from torchvision import transforms as torchtrans
import os
import wandb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
import copy
import tqdm
# Faster R-CNN model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Model evalutation
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# For image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



# Package versions
#print(f"{os.popen('python --version').read().strip()}")
#print(f"OpenCV version: {cv2.__version__}")
#print(f"Numpy version: {np.__version__}")
#print(f"wandb version: {wandb.__version__}")
#print("Albumentations version:", A.__version__)
#print(f"Matplotlib version: {plt.matplotlib.__version__}")
#print(f"Torchmetrics version: {torchmetrics.__version__}")
#print(f"Pytorch version: {torch.__version__}")
#print(f"Torchvision version: {torchvision.__version__}")
#print(f"CUDA available: {torch.cuda.is_available()}")
#print(f"CUDA version: {torch.version.cuda}")

import sys
from utils.utils import *
from utils.transforms import *
from utils.coco_eval import *
from utils.engine import *
from utils.coco_utils import *



def yolo_to_box(yolo_box, W, H):
    """
    Helper funtion to transform the bounding box from YOLO format to pascal VOC format.
    From: YOLO format: [x_center, y_center, width, height]
    To: Pascal VOC format: [x1, y1, x2, y2]
    """
    x_center, y_center, width, height = yolo_box

    # Convert normalized coordinates to absolute coordinates
    x1 = (x_center - width / 2) * W
    y1 = (y_center - height / 2) * H
    x2 = (x_center + width / 2) * W
    y2 = (y_center + height / 2) * H

    # Ensure x1, y1, x2, y2 are within valid ranges
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))

    return [x1, y1, x2, y2]

class FashionDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for the fashion dataset.
    Loads annotations from YOLO format, but returns then in pascal voc format.

    """
    def __init__(self, data_folder, split, width, height, transform=None):
        self.split = split
        self.data_folder = data_folder
        self.transform = transform
        self.width = width
        self.height = height

        # Get a list of image and labels paths
        img_names = list(os.listdir(os.path.join(data_folder, 'images', split)))
        self.img_pth = [os.path.join(data_folder, 'images', split, img) for img in img_names]
        self.label_pth = [os.path.join(data_folder, 'labels', split, img.split('.')[0] + '.txt') for img in img_names]
        
    def __len__(self):
        return len(self.img_pth)
    
    def __getitem__(self, idx):
        img_pth = self.img_pth[idx]
        label_pth = self.label_pth[idx]

        # Load image
        img = cv2.imread(img_pth)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res /= 255.0

        # load labels
        labels = []
        boxes = []
        with open(label_pth, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                label = int(parts[0]) + 1 # 0 is background, all dataset classes must be +1
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                labels.append(label)
                yolo_format  = yolo_to_box([x_center, y_center, width, height], self.width, self.height)
                boxes.append(yolo_format)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Transformations
        if self.transform:
            sample = self.transform(image = img_res,
                                     bboxes = boxes,
                                     labels = labels)
        # Conver tot tensor for validation data (or no augmentations)
        else:
            transform = A.Compose([
                ToTensorV2(p=1.0)
                ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
            sample = transform(image = img_res,
                               bboxes = boxes,
                               labels = labels)
            
        img_res = sample['image']
        boxes = torch.Tensor(sample['bboxes'])

        if len(boxes)==0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": int(idx),
                "id": idx,
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
        else:
            target = {
                "boxes":boxes, "labels":labels,
                "image_id":int(idx),
                "id":idx, # TODO: maybe remove?
                "area":(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd":torch.zeros((len(boxes),), dtype=torch.int64)}
        return img_res, target


def plot_img_bbox(img, target):
    """
    Plots image (img) with boxes (target).
    Target should be a list with boxes, boxes should be in for format [x1,y1,x2,y2]
    """
    _, h, w = img.shape
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img.permute(1,2,0))
    for box in (target['boxes']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 0.5,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()

def get_object_detection_model(num_classes, trainable_backbone_layers=3, box_score_thresh=0.05, box_nms_thresh=0.5):
    """
    num_classes: number of classes in dataset (including background)
    trainable_backbone_layers: number of trainable (not frozen) layers starting from final block. 0-5, 5 = all. default=3
    box_score_thresh (float): during inference, only return proposals with a classification score greater than box_score_thresh
    box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
    """

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1',
                                                                 trainable_backbone_layers=trainable_backbone_layers,
                                                                 box_score_thresh=box_score_thresh,
                                                                 box_nms_thresh=box_nms_thresh)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def apply_nms(orig_prediction, iou_thresh=0.3):
    final_prediction = copy.copy(orig_prediction)

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(final_prediction['boxes'], final_prediction['scores'], iou_thresh)
    
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

def apply_nms_per_class(orig_prediction, iou_thresh=0.3):
    final_prediction = copy.deepcopy(orig_prediction)
    boxes = final_prediction['boxes']
    scores = final_prediction['scores']
    labels = final_prediction['labels']
    
    # Initialize lists to store filtered boxes, scores, and labels
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    
    # Get unique class labels
    unique_labels = [1,2,3] #torch.unique(labels)
    
    # Apply NMS for each class separately
    for label in unique_labels:
        # Filter boxes, scores, and labels by class
        class_indices = (labels == label).nonzero().squeeze(1)
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        
        # Apply NMS for the current class
        keep = torchvision.ops.nms(class_boxes, class_scores, iou_thresh)
        
        # Append filtered boxes, scores, and labels
        filtered_boxes.append(class_boxes[keep])
        filtered_scores.append(class_scores[keep])
        filtered_labels.append(torch.full_like(keep, label))
    
    # Concatenate filtered boxes, scores, and labels for all classes
    final_prediction['boxes'] = torch.cat(filtered_boxes)
    final_prediction['scores'] = torch.cat(filtered_scores)
    final_prediction['labels'] = torch.cat(filtered_labels)
    
    return final_prediction

# Construct model name
from datetime import datetime
def get_mod_name(model, name=None):
    """
    Constructs model name with date and time.
    """
    # get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    if name:
        return f"{timestamp}_FasterRCNN_{name}"
    else:
        return f"{timestamp}_FasterRCNN"
    
class EarlyStoppingAP:
    """
    Early stopping class, monitoring validation mAP
    """
    def __init__(self, patience=5):
        self.higest_ap = -np.inf
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, ap):
        if ap < self.higest_ap:
            self.counter +=1
            print(f"Val AP did not improve, current: {ap}, best: {self.higest_ap} count: {self.counter}/{self.patience}\n")
            if self.counter >= self.patience:  
                self.early_stop = True
        else:
            print(f"AP improved! from {self.higest_ap}, to: {ap}, Saving model weights :)\n")
            self.higest_ap = ap
            self.counter = 0


class EarlyStoppingLoss:
    """
    Early stopping class, monitoring validation loss
    """
    def __init__(self, patience=5):
        self.lowest_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss > self.lowest_loss:
            self.counter +=1
            print(f"Val loss did not improve, current: {validation_loss}, best: {self.lowest_loss} count: {self.counter}/{self.patience}\n")
            if self.counter >= self.patience:  
                self.early_stop = True
        else:
            print(f"Val loss improved! from {self.lowest_loss}, to: {validation_loss}, Saving model weights :)\n")
            self.lowest_loss = validation_loss
            self.counter = 0

def get_ap(model, data_loader, device, nms_tresh=None):
    """
    Computes mAP@[.5,.95,.05] and mAP@50
    """
    coco_eval_obj = utils.cocoeval.evaluate(model, data_loader, device, nms_tresh)
    metrics = coco_eval_obj.coco_eval["bbox"].stats
    ap_50_95 = metrics[0]
    ap_50 = metrics[1]
    return ap_50_95, ap_50

def get_val_loss(model, val_loader, device, print_freq=1000):
    """
    Compute model loss.
    """
    model.train()
    metric_logger = utils.utils.MetricLogger(delimiter="  ")
    header = "Validation:"

    with torch.no_grad():
        for images, targets in metric_logger.log_every(val_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            loss_dict_reduced = utils.utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

    return metric_logger

############################ Evaluation #########################
def apply_threshold(orig_prediction, threshold):
    """
    Apply confidence threshold (orig_prediction) to the predictions (orig_prediction)
    """
    # Filter predictions based on confidence score threshold
    mask = orig_prediction['scores'] >= threshold
    filtered_scores = orig_prediction['scores'][mask]
    filtered_boxes = orig_prediction['boxes'][mask]
    filtered_labels = orig_prediction['labels'][mask]
    
    # Create a new prediction dictionary
    filtered_prediction = {
        'scores': filtered_scores,
        'boxes': filtered_boxes,
        'labels': filtered_labels
    }
    return filtered_prediction

class ValidationMetric():
    """
    Wrapper class for extracting ground truths, predicted value and evaluation metric.
    """
    def __init__(self, model, dataloader, device, nms_tresh=None, conf_tresh=None, iou_thresholds=None):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.nms_tresh = nms_tresh
        self.conf_tresh = conf_tresh
        self.metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds=iou_thresholds)
    
    def get_prediction(self):
        self.model.eval()
        all_predictions = []
        for images, _ in self.dataloader:
            with torch.no_grad():
                images = list(image.to(self.device) for image in images)
                predictions = self.model(images)
                if self.nms_tresh:
                    predictions = [apply_nms(pred, iou_thresh=self.nms_tresh) for pred in predictions]
                if self.conf_tresh:
                    predictions = [apply_threshold(pred, threshold=self.conf_tresh) for pred in predictions]
                all_predictions.extend(predictions)
        return all_predictions
    
    def get_ground_truth(self):
        all_ground_truth = []
        for _, targets in self.dataloader:
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            all_ground_truth.extend(targets)
        return all_ground_truth
    
    def remove_tensor(self, res):
        res_no_tensors = {}
        for key, value in res.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    res_no_tensors[key] = value.item()
                else:
                    res_no_tensors[key] = value.tolist()
            else:
                res_no_tensors[key] = value
        return res_no_tensors
    
    def compute_mAP(self):
        predictions = self.get_prediction()
        ground_truth = self.get_ground_truth()
        self.metric.update(predictions, ground_truth)
        result = self.metric.compute()
        return self.remove_tensor(result)
    
    def change_all_preds(self, obj):
        for o in obj:
            labels_length = o["labels"].shape[0]
            new_labels = torch.ones(labels_length, dtype=torch.int64, device=o["labels"].device)
            o["labels"] = new_labels
        return obj
    
    def compute_mAP_noclass(self):
        predictions = self.get_prediction()
        predictions = self.change_all_preds(predictions)
        ground_truth = self.get_ground_truth()
        ground_truth = self.change_all_preds(ground_truth)
        self.metric.update(predictions, ground_truth)
        result = self.metric.compute()
        return self.remove_tensor(result)


def plot_img_bbox_pred(img, target, CLASSES, predicted_target=None, save_path=None):
    """
    Plots image with predicted bboxes
    """
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(30,30)
    a.imshow(img.permute(1,2,0))
    a.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    
    class_colors = {
        1: 'r',  # Class 0: Red
        2: 'g',  # Class 1: Green
        3: 'b'   # Class 2: Blue
    }

    # Plot true boxes
    for box, label in zip(target['boxes'], target['labels']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=1.5,
                                 edgecolor=class_colors[label.item()],
                                 linestyle='dotted',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
        # Add class score label
        # a.text(x, y, f"{CLASSES[label.item()]}", color=class_colors[label.item()], fontsize=10, verticalalignment='top')
    
    # Plot predicted boxes if provided
    if predicted_target:
        for box, label, score in zip(predicted_target['boxes'], predicted_target['labels'], predicted_target['scores']):
            x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth=1.5,
                                     edgecolor=class_colors[label.item()],
                                     facecolor='none')

            # Draw the bounding box on top of the image
            a.add_patch(rect)
            # Add class score label
            # Adjust the vertical position of text for predicted boxes
            a.text(x+(width/2)-20, y - 11, f"{CLASSES[label.item()]} {score:.2f}", color=class_colors[label.item()], fontsize=15, verticalalignment='top')

    print(f"Predicted: {len(predicted_target['labels']) if predicted_target else 0} boxes")
    print(f"Actual: {len(target['labels'])} boxes")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
    else:
        return fig


def predict_on_valid(model, val_dataset, CLASSES, idx, nms_tres=None, conf_tres=None):
    """
    Inference on validation dataloader.
    Plots the sample with index idx.
    """
    # Assuming your model is already on GPU
    model.cuda()

    # Assuming train_dataset is your dataset
    img, target = val_dataset[idx]

    # Move input image to GPU
    img = img.cuda()

    # Predict using model
    model.eval()
    with torch.no_grad():
        # Move the input image to the same device as the model
        pred_target = model(img.unsqueeze(0).cuda())[0]
    if nms_tres:
        pred_target = apply_nms(pred_target, nms_tres)
    if conf_tres:
        pred_target = apply_threshold(pred_target, conf_tres)

    # Move predicted target to CPU for plotting
    pred_target_cpu = {key: value.cpu() for key, value in pred_target.items()}

    # Plot the image with predicted bounding boxes
    return plot_img_bbox_pred(img.cpu(), target, CLASSES, pred_target_cpu)


# Predict on sample
# loaded directly from folder
def load_sample(data_folder, split, img_name):
    """
    Perform inference in a sample with image name (img_name) from
    datafolder (data_folder) and datasplit (split).

    """
    img_pth = os.path.join(data_folder, 'images', split, img_name)
    label_pth = os.path.join(data_folder, 'labels', split, img_name.split('.')[0] + '.txt')

    img = cv2.imread(img_pth)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_res = cv2.resize(img_rgb, (w, h), cv2.INTER_AREA)
    img_res /= 255.0
    # load labels
    labels = []
    boxes = []
    with open(label_pth, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            label = int(parts[0]) + 1 # 0 is background, all dataset classes must be +1
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            labels.append(label)
            yolo_format  = yolo_to_box([x_center, y_center, width, height], w, h)
            boxes.append(yolo_format)
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    transform = A.Compose([ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    sample = transform(image = img_res, bboxes = boxes, labels = labels)
        
    img_res = sample['image']
    boxes = torch.Tensor(sample['bboxes'])

    if len(boxes)==0:
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64)
        }
    else:
        target = {
            "boxes":boxes, "labels":labels,
            "area":(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd":torch.zeros((len(boxes),), dtype=torch.int64)}
    return img_res, target

def predict_on_image(model, data_folder, split, sample_id, CLASSES, nms_tres=None, conf_tres=None, save_path=None):
    """
    Plots the image and bboxes.
    """
    img, target = load_sample(data_folder, split, sample_id)
    img = img.cuda()
    model.eval()
    pred_target = model(img.unsqueeze(0))
    with torch.no_grad():
        # Move the input image to the same device as the model
        pred_target = model(img.unsqueeze(0).cuda())[0]
    if nms_tres:
        pred_target = apply_nms(pred_target, nms_tres)
    if conf_tres:
        pred_target = apply_threshold(pred_target, conf_tres)

    pred_target_cpu = {key: value.cpu() for key, value in pred_target.items()}
    return plot_img_bbox_pred(img.cpu(), target, CLASSES, pred_target_cpu,save_path=save_path)


### f1 score

def get_iou(a, b, epsilon=1e-5, intersection_check=False):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        if intersection_check:
            return 0.0, False
        else:
            return 0.0
    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)
    if intersection_check:
        return iou, bool(area_overlap)
    else:
        return iou
    
def get_object_size(box):
    # Calculate the area of the bounding box
    width = box[2] - box[0]
    height = box[3] - box[1]
    area = width * height
    
    # Determine the object size category based on area
    if area < 32**2:
        return 'small'
    elif area >= 32**2 and area < 96**2:
        return 'medium'
    else:
        return 'large'

def calc_conditions_class(gt_boxes, pred_boxes, gt_labels, pred_labels, iou_thresh=0.5):
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}
    class_index = [1, 2, 3]  # DO NOT CHANGE THIS LINE!
    matched_gt_indices = set()  # Keep track of matched ground truth boxes to avoid double counting
    matched_pred_indices = set()  # Keep track of matched predictions to avoid double counting

    for class_id in class_index:
        tp_dict[class_id] = 0
        fp_dict[class_id] = 0
        fn_dict[class_id] = 0

    # Iterate over ground truth boxes
    for i, gt_box in enumerate(gt_boxes):
        max_iou = -1  # Initialize maximum IoU
        max_iou_pred_index = -1  # Index of the prediction with maximum IoU
        for j, pred_box in enumerate(pred_boxes):
            now_iou, intersect = get_iou(gt_box, pred_box, intersection_check=True)
            if intersect and pred_labels[j] == gt_labels[i]:
                if now_iou >= iou_thresh and now_iou > max_iou:
                    max_iou = now_iou
                    max_iou_pred_index = j

        # If a matching prediction is found, count it as a true positive
        if max_iou_pred_index != -1 and max_iou_pred_index not in matched_pred_indices:
            tp_dict[gt_labels[i]] += 1
            matched_gt_indices.add(i)
            matched_pred_indices.add(max_iou_pred_index)

    # Count unmatched ground truth boxes as false negatives
    for i, gt_box in enumerate(gt_boxes):
        if i not in matched_gt_indices:
            fn_dict[gt_labels[i]] += 1

    # Count unmatched predicted boxes as false positives
    for j, pred_box in enumerate(pred_boxes):
        if j not in matched_pred_indices:
            label = pred_labels[j]
            fp_dict[label] += 1

    return tp_dict, fp_dict, fn_dict


def calc_conditions_size(gt_boxes, pred_boxes, gt_labels, pred_labels, iou_thresh=0.5):
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}
    size_categories = ['small', 'medium', 'large']  # Define size categories
    matched_gt_indices = set()
    matched_pred_indices = set()

    for size_cat in size_categories:
        tp_dict[size_cat] = 0
        fp_dict[size_cat] = 0
        fn_dict[size_cat] = 0

    # Iterate over ground truth boxes
    for i, gt_box in enumerate(gt_boxes):
        max_iou = -1  # Initialize maximum IoU
        max_iou_pred_index = -1  # Index of the prediction with maximum IoU
        gt_size = get_object_size(gt_box)  # Determine size category of ground truth box

        # Find the prediction with the highest IoU for the current ground truth box
        for j, pred_box in enumerate(pred_boxes):
            now_iou, intersect = get_iou(gt_box, pred_box, intersection_check=True)
            if intersect and pred_labels[j] == gt_labels[i]:
                if now_iou >= iou_thresh and now_iou > max_iou:
                    max_iou = now_iou
                    max_iou_pred_index = j

        # If a matching prediction is found, count it as a true positive
        if max_iou_pred_index != -1 and max_iou_pred_index not in matched_pred_indices:
            tp_dict[gt_size] += 1
            matched_gt_indices.add(i)
            matched_pred_indices.add(max_iou_pred_index)

    # Count unmatched ground truth boxes as false negatives
    for i, gt_box in enumerate(gt_boxes):
        if i not in matched_gt_indices:
            size_cat = get_object_size(gt_box)
            fn_dict[size_cat] += 1

    # Count unmatched predicted boxes as false positives
    for j, pred_box in enumerate(pred_boxes):
        if j not in matched_pred_indices:
            size_cat = get_object_size(pred_box)
            fp_dict[size_cat] += 1

    return tp_dict, fp_dict, fn_dict

def calculate_tp_fp_fn_one_image(preds, target, iou_threshold=0.5, type="class"):
    true_positives = {}
    false_positives = {}
    false_negatives = {}
    
    for pred in preds:
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        
        for t in target:
            target_boxes = t['boxes']
            target_labels = t['labels']
            
            if type == "class":
                tp, fp, fn = calc_conditions_class(target_boxes, pred_boxes, target_labels, pred_labels, iou_threshold)
            elif type == "size":
                tp, fp, fn = calc_conditions_size(target_boxes, pred_boxes, target_labels, pred_labels, iou_threshold)
            
            for size_cat in tp:
                if size_cat not in true_positives:
                    true_positives[size_cat] = 0
                true_positives[size_cat] += tp[size_cat]
            
            for size_cat in fp:
                if size_cat not in false_positives:
                    false_positives[size_cat] = 0
                false_positives[size_cat] += fp[size_cat]
            
            for size_cat in fn:
                if size_cat not in false_negatives:
                    false_negatives[size_cat] = 0
                false_negatives[size_cat] += fn[size_cat]
    
    return true_positives, false_positives, false_negatives

def calculate_tp_fp_fn(preds, target, iou_threshold=0.5, type="class"):
    total_tp = {}
    total_fp = {}
    total_fn = {}
    for pred, target_ in zip(preds, target):
        tp, fp, fn = calculate_tp_fp_fn_one_image([pred], [target_], iou_threshold=iou_threshold, type=type)
        for size_cat in tp:
            if size_cat not in total_tp:
                total_tp[size_cat] = 0
            total_tp[size_cat] += tp[size_cat]
        
        for size_cat in fp:
            if size_cat not in total_fp:
                total_fp[size_cat] = 0
            total_fp[size_cat] += fp[size_cat]
        
        for size_cat in fn:
            if size_cat not in total_fn:
                total_fn[size_cat] = 0
            total_fn[size_cat] += fn[size_cat]

    return total_tp, total_fp, total_fn


def calculate_f1(predictions, ground_truth, iou_threshold=0.5):
    res = calculate_tp_fp_fn(predictions, ground_truth, iou_threshold=iou_threshold)
    total_tp = sum(res[0].values())
    total_fp = sum(res[1].values())
    total_fn = sum(res[2].values())
    #print(f"Total True Positives: {total_tp}, Total False Positives: {total_fp}, Total False Negatives: {total_fn}")
    # Calculate precision, recall and F1 score
    try:
        precision = total_tp / (total_tp + total_fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = total_tp / (total_tp + total_fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return precision, recall, f1

########## Tune thresholds ##########
## Helper function
def transform_target(dict_):
    res = dict(
        boxes=dict_['boxes'].detach().cpu().numpy(),
        labels=dict_['labels'].detach().cpu().numpy()
    )
    return res

# Function to plot a heatmap of the results
def plot_res(res, best_type="min",save_path=None):
    nmn_list = np.round(np.arange(0, 0.55, 0.05), 2)
    conf_list = np.round(np.arange(0.05, 1.05, 0.05), 2)
    # Create a heatmap
    plt.figure(figsize=(14, 6))  # Adjust the figure size as needed
    plt.imshow(res, cmap='coolwarm', interpolation='nearest')

    # Set labels for x and y axis
    plt.xticks(np.arange(len(conf_list)), conf_list, fontsize=11)
    plt.yticks(np.arange(len(nmn_list)), nmn_list, fontsize=11)

    # Add axis labels
    plt.xlabel('Confidence threshold', fontsize =15)
    plt.ylabel('NMS threshold', fontsize =15)

    
    cbar = plt.colorbar(label='F1-score')
    cbar.ax.tick_params(labelsize=13)  # Set font size for color bar tick labels
    cbar.set_label('F1-score', fontsize=15) 

    if best_type=="max":
        # Find indices of the highest value
        best_idx = np.argwhere(res == np.max(res))
    else:
        best_idx = np.argwhere(res == np.min(res))

    plt.rcParams.update({'font.size': 13})
    # Plot red dots for the highest values
    for i, idx in enumerate(best_idx):
        if i == 0:
            plt.scatter(idx[1], idx[0], color='black', s=70, label=f"Highest F1-score: {round(res[idx[0], idx[1]],3)}")
        else:
            plt.scatter(idx[1], idx[0], color='black', s=70)


    # Add a title
    plt.title('F1-score heatmap', fontsize=17)

    # Show plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.1))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.show()
#  Function to test diffrent threshodls
def opt_thresholds(model, data_loader, device, iou_threshold=0.25):
    """
    Optimal thresholds by mAP50
    """
    met = ValidationMetric(model, data_loader, device)
    gt = met.get_ground_truth()
    gt = [transform_target(target) for target in gt]
    preds = met.get_prediction()

    print("All set up!")

    best_f1 = -1  # Initialize best mAP score
    best_nms = None  # Initialize best nms threshold
    best_conf = None  # Initialize best confidence threshold

    def test_thres(nms, conf, iou_threshold):
        predictions = copy.copy(preds)
        predictions = [apply_nms(pred, iou_thresh=nms) for pred in predictions]
        predictions = [apply_threshold(pred, threshold=conf) for pred in predictions]
        predictions = [transform_target(p) for p in predictions]
        precision, recall, f1 =  calculate_f1(predictions, gt, iou_threshold=iou_threshold)
        return f1
    
    nmn_list = np.round(np.arange(0, 0.55, 0.05), 2)
    conf_list = np.round(np.arange(0.05, 1.05, 0.05), 2)
    res = np.zeros((len(nmn_list), len(conf_list)))
    print(f"Running {len(nmn_list)*len(conf_list)} thresholds")
    c = 0
    for i, nmn in tqdm.tqdm(enumerate(nmn_list), total=len(nmn_list)):
        for j, conf in enumerate(conf_list):
            c += 1
            f1 = test_thres(nmn, conf, iou_threshold)
            res[i, j] = f1
            #print(f"Counter: {c}, nms:{nmn}, conf:{conf}, f1:{f1}")
            # Check if current mAP is better than the previous best
            if f1 > best_f1:
                best_f1 = f1
                best_nms = nmn
                best_conf = conf
            
    print(f"Best f1: {best_f1}")
    print(f"Best NMS Threshold: {best_nms}")
    print(f"Best Confidence Threshold: {best_conf}")
            
    return res

