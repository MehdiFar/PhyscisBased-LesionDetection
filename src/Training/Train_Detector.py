import os
import __init__
import numpy as np
from typing import Optional
import torch
from torch import nn
from PIL import Image, ImageDraw
import pandas as pd
import utils
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import torchvision
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FastRCNNConvFCHead, FasterRCNN_ResNet50_FPN_Weights
import transforms as T
import dicaugment as dca
from engine import train_one_epoch, evaluate
import utils
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models import resnet50, ResNet50_Weights
import random


from tqdm import tqdm
import datetime
from pathlib import Path
import csv
from torchinfo import summary
import evaluation
import resource
from config import config


def read_csv_gt(csv_dir):
	df = pd.read_csv(csv_dir)
	gt_list = []
	for idx, row in df.iterrows():
		volume_name = "_".join(row['File_name'].split("_")[:-1])
		volume_name = volume_name + '_' + str(row['Key_slice_index']).zfill(4)
		slice_index = row['Key_slice_index']
		bbox = row['Bounding_boxes']
		lesion_type = row['Coarse_lesion_type']
		Possibly_noisy = row['Possibly_noisy']
		spacing = float(row['Spacing_mm_px_'].split(',')[0])

		
		Image_size = row['Image_size']
		if Possibly_noisy == 1: 
			continue
		if Image_size.split(',')[0] not in ['512', ' 512', '512 '] or Image_size.split(',')[1] not in ['512', ' 512', '512 ']:
			continue
		
		y1 = max(1.0, float(bbox.split(',')[0]))
		y2 = min(511.0, float(bbox.split(',')[2]))
		x1 = max(1.0, float(bbox.split(',')[1]))
		x2 = min(511.0, float(bbox.split(',')[3]))

		gt_list.append([volume_name, y1, x1, y2, x2, lesion_type, spacing])
	return gt_list

class DeepLesion(torch.utils.data.Dataset):
	def __init__(self, root, gt_csv, transforms, transform):
		self.root = root
		self.transforms = transforms
		self.gt_csv = gt_csv
		self.imgs = list(sorted(os.listdir(root)))
		self.gt_annotations = read_csv_gt(gt_csv)
		self.transform = transform

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, self.imgs[idx])
		image_filename = img_path.split('/')[-1][:-4]
		img = np.load(img_path)

		metaData = image_filename.split('_')
		img_slice_name = "_".join(metaData[:4])
		annots = [x for x in self.gt_annotations if x[0] == img_slice_name]
		boxes = []
		labels = []
		image_id = []



		for annot in annots:
			x1 = np.round(float(annot[1]))
			y1 = np.round(float(annot[2]))
			x2 = np.round(float(annot[3]))
			y2 = np.round(float(annot[4]))
			boxes.append([x1, y1, x2, y2])
			labels.append(1)
		
		if self.transform:
			for box in boxes:
				box.insert(2,0)
				box.insert(5,2)
			kernel_list = ['b10f', 'b20f', 'b22f', 'b26f', 'b30f', 'b31f', 'b35f', 'b36f', 'b40f', 
				  		   'b41f', 'b43f', 'b45f', 'b46f', 'b50f', 'b60f', 'b70f', 'b75f', 'b80f', 
				  		   'bone', 'boneplus', 'chest', 'detail', 'edge', 'lung', 'soft', 'standard']
			dicom = {
				"PixelSpacing" : (annots[0][6], annots[0][6]),
				"RescaleIntercept" : 0,
				"RescaleSlope" : 1.0,
				"ConvolutionKernel" : random.choice(kernel_list),
				"XRayTubeCurrent" : 240
				}
			transformed = self.transform(image=img, bboxes = boxes, class_labels = labels, dicom = dicom)
			img = transformed["image"]
			boxes = transformed["bboxes"]
			boxes = [list(item) for item in boxes]
			for box in boxes:
				del box[2]
				del box[-1]

		img = np.clip(img, -1024, 3071)
		img = ((img+1024)/(3071+1024))*255.0
		img = Image.fromarray(img.astype('uint8'),'RGB')
		
		
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.as_tensor(labels, dtype=torch.int64)
		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		iscrowd = torch.zeros((1,), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target
	
	def __len__(self):
		return len(self.imgs)
	
	

class DeepLesion_test(torch.utils.data.Dataset):
	def __init__(self, root, transforms):
		self.root = root
		self.transforms = transforms
		self.imgs = list(sorted(os.listdir(root)))

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, self.imgs[idx])
		image_filename = img_path.split('/')[-1][:-4]
		img = np.load(img_path)
		img = np.clip(img, -1024, 3071)
		img = ((img+1024)/(3071+1024))*255.0

		img = Image.fromarray(img.astype('uint8'),'RGB')

		metaData = image_filename.split('_')
		lesionType = int(metaData[5])
		boxes = []
		boxes.append([float(metaData[7]), float(metaData[9]), float(metaData[11]), float(metaData[13])])

		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.ones((1,), dtype=torch.int64)


		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		iscrowd = torch.zeros((1,), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target, self.imgs[idx]

	def __len__(self):
		return len(self.imgs)
	

def build_model(hyper_params):
	model_backbone = hyper_params["model_backbone"]
	box_score_trhesh = hyper_params["box_score_thresh"]
	rpn_fg_iou_thresh = hyper_params["rpn_fg_iou_thresh"]
	rpn_bg_iou_thresh = hyper_params["rpn_bg_iou_thresh"]
	box_fg_iou_thresh = hyper_params["box_fg_iou_thresh"]
	box_bg_iou_thresh = hyper_params["box_bg_iou_thresh"]
	if model_backbone == 'fasterrcnn_resnet50_fpn':
		model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT",
															   		box_score_thresh = box_score_trhesh,
																	rpn_fg_iou_thresh = rpn_fg_iou_thresh,
																	rpn_bg_iou_thresh = rpn_bg_iou_thresh,
																	box_fg_iou_thresh = box_fg_iou_thresh,
																	box_bg_iou_thresh = box_bg_iou_thresh)
		num_classes = 2  # 1 class (lesion) + background
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	else:
		anchor_aspect_ratio = hyper_params["anchor_aspect_ratio"]
		anchor_size = hyper_params["anchor_size"]
		trainable_backbone_layers = hyper_params["trainable_backbone_layers"]			
		if model_backbone == 'resnet':
			weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
			weights_backbone = None
			weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
			weights_backbone = ResNet50_Weights.verify(weights_backbone)

			if weights is not None:
				weights_backbone = None
				num_classes = _ovewrite_value_param("num_classes", 91, len(weights.meta["categories"]))
			elif num_classes is None:
				num_classes = 91

			is_trained = weights is not None or weights_backbone is not None
			trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
			norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

			backbone = resnet50(weights=weights_backbone, progress=True, norm_layer=norm_layer)
			backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

			anchor_generator = AnchorGenerator(sizes=anchor_size,aspect_ratios=anchor_aspect_ratio)
			roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        												featmap_names=['0'],
        												output_size=hyper_params["roipooler_outputSize"],
        												sampling_ratio=hyper_params["roipooler_sampling_ratio"],
														canonical_scale=hyper_params["roipooler_canonical_scale"],
														canonical_level=hyper_params["roipooler_canonical_level"]
    													)

			model = FasterRCNN(backbone=backbone,
							num_classes=num_classes,
							rpn_head = None,
							box_head = None,
							box_score_thresh = box_score_trhesh,
							rpn_fg_iou_thresh = rpn_fg_iou_thresh,
							rpn_bg_iou_thresh = rpn_bg_iou_thresh,
							box_fg_iou_thresh = box_fg_iou_thresh,
							box_bg_iou_thresh = box_bg_iou_thresh,
							rpn_anchor_generator=anchor_generator,
							box_roi_pool=roi_pooler
							)

			if weights is not None:
				model.load_state_dict(weights.get_state_dict(progress=True))
				if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
					overwrite_eps(model, 0.0)

			in_features = model.roi_heads.box_predictor.cls_score.in_features
			model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
			return model

		
	return model


def get_transform():
	transforms = []
	transforms.append(T.PILToTensor())
	transforms.append(T.ConvertImageDtype(torch.float))
	return T.Compose(transforms)


def get_transform_dicaugment(aug_policy):

	return dca.Compose([
		dca.NPSNoise(magnitude=aug_policy['NPS_params'][0], p = aug_policy['NPS_params'][1]),
		dca.OneOf([
			dca.Blur(blur_limit = aug_policy['Blur'][0], by_slice=aug_policy['Blur'][1], mode = aug_policy['Blur'][2], cval = aug_policy['Blur'][3], p = aug_policy['Blur'][4]),
			dca.Sharpen(alpha = aug_policy['Sharpen'][0], lightness=aug_policy['Sharpen'][1], mode = aug_policy['Sharpen'][2], cval = aug_policy['Sharpen'][3], p = aug_policy['Sharpen'][4]),
			], p=aug_policy["oneOF"][0]),
		dca.OneOf([
			dca.HorizontalFlip(p = aug_policy['HorizontalFlip']),
			dca.VerticalFlip(p = aug_policy['VerticalFlip']),
			dca.RandomRotate90(axes=aug_policy['RandomRotate90'][0], p = aug_policy['RandomRotate90'][1]),
			], p=aug_policy["oneOF"][1])
		], bbox_params= dca.BboxParams(format='pascal_voc_3d', label_fields=['class_labels']), p=aug_policy["oneOF"][2])



def write_detection_csvFile(model, data_loader, log_dir, device):
	n_threads = torch.get_num_threads()
	torch.set_num_threads(1)
	cpu_device = torch.device("cpu")
	model.eval()
	stream = tqdm(data_loader, total=len(data_loader))

	with torch.no_grad():
		with open(os.path.join(log_dir,'detection.csv'), 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			writer.writerow(['uid', 'x_min', 'y_min', 'x_max', 'y_max', 'scores'])


			for i, (images, targets, uids) in enumerate(stream, start=1):
				images = list(img.to(device) for img in images)

				if torch.cuda.is_available():
					torch.cuda.synchronize()
				outputs = model(images)

				outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

				for idx in range(len(images)):
					image_names = []
					x_min, y_min, x_max, y_max = [], [], [], []
					probability = []

					pred = outputs[idx]
					img = images[idx]
					img = (255.0 * (img - img.min()) / (img.max() - img.min())).to(torch.uint8)
					img = img[:3, ...]
					bbox = pred['boxes'].cpu().numpy()
					score = pred['scores'].cpu().numpy()
					if bbox.shape[0] == 0:
						continue
					uid = uids[idx].split("_")
					uid = uid[0] + "_" + uid[1] + "_" + uid[2] + "_" + uid[3]
					for b in range(bbox.shape[0]):
						image_names.append(uid)
						x_min.append(bbox[b][0])
						y_min.append(bbox[b][1])
						x_max.append(bbox[b][2])
						y_max.append(bbox[b][3])
						probability.append(score[b])
					rows = zip(image_names, x_min, y_min, x_max, y_max, probability)
					for row in rows:
						writer.writerow(row)



if __name__ == "__main__":
	hyper_params = config().params

	model = build_model(hyper_params)
	log_dir = os.path.join(hyper_params["saved_models_dir"], str(datetime.datetime.now()).replace(" ", "_"))
	log_dir = log_dir.replace(":", "-")
	load_data_dir = hyper_params["data_dir"]
	
	device = torch.device(hyper_params["device"]) if torch.cuda.is_available() else torch.device('cpu')

	num_classes = 2
	dataset = DeepLesion(os.path.join(load_data_dir,'Train'), hyper_params["gt_annotations_dir"], 
					   get_transform(), get_transform_dicaugment(aug_policy=hyper_params["aug_policy"]))
	

	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=hyper_params["batch_size"], shuffle=True, num_workers=hyper_params["num_workers"],
		collate_fn=utils.collate_fn)


	model.to(device)

	if hyper_params["unfreeze"]:
		params = [p for p in model.parameters()]
	else:
		params = [p for p in model.parameters() if p.requires_grad]

	if hyper_params["optimizer"] == 'sgd':
		optimizer = torch.optim.SGD(params, lr=hyper_params["lr"],
							momentum=hyper_params["momentum"], weight_decay=hyper_params["weight_decay"])
		
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
										   step_size=hyper_params["step_size"],
										   gamma=hyper_params["gamma"])

	num_epochs = hyper_params["epochs"]

	for epoch in range(num_epochs):
		train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
		lr_scheduler.step()
	
	if not os.path.exists(log_dir):
		Path(log_dir).mkdir(parents=True,exist_ok=True)

	torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))
	df = pd.DataFrame.from_dict(hyper_params, orient = 'index')
	df.to_csv(os.path.join(log_dir, "HypeParameters.csv"))
	

	test_final = DeepLesion_test(os.path.join(load_data_dir, 'Tune_Evaluation'), get_transform())
	data_loader_final = torch.utils.data.DataLoader(test_final, batch_size=4, shuffle=False, num_workers=hyper_params["num_workers"],
	collate_fn=utils.collate_fn)
	write_detection_csvFile(model, data_loader_final, log_dir= log_dir, device=device)


	pred_csv_dir = os.path.join(log_dir, 'detection.csv')
	evaluation.FROC_Evaluation(hyper_params["gt_annotations_dir"], pred_csv_dir = pred_csv_dir, log_dir = log_dir, partition = 2)
	evaluation.compute_CPM(log_dir)

	print("Training and Evaluation are finished and the results are stored in " + log_dir)




