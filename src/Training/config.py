class config(object):
	def __init__(self, params = {}):
		self.params = {}
		self.params["num_workers"] = 0
		self.params["batch_size"] = 4
		self.params["device"] = "cpu" # cuda:0, cpu

		self.params["optimizer"] = 'sgd'
		self.params["epochs"] = 15
		self.params["lr"] = 0.01	# 0.01, 0.02, 0.04, 0.08, 0.005, 0.001, 0.008, 0.1, 0.03, 0.2
		self.params["gamma"] = 0.1	# 0.5, 0.1
		self.params["momentum"] = 0.9  # 0.9, 0.7, 0.75, 0.8, 0.85, 0.95, 0.98
		self.params["weight_decay"] = 0.0002 # 0.0005, 0.001, 0.0001,   0.00001, 0.00005, 0.000005, 0.0002
		self.params["step_size"] = 3	# 3, 5 [3, 7, 9]

		self.params["model_backbone"] = "resnet" # fasterrcnn_resnet50_fpn, squeezenet, resnet, vgg, Inception
		self.params["box_score_thresh"] = 0.0001
		self.params["rpn_fg_iou_thresh"] = 0.8	# 0.7, 0.5, 0.3, 0.85, 0.75, 	0.90, 0.95
		self.params["rpn_bg_iou_thresh"] = 0.2	# 0.3, 0.5, 0.1, 0.05
		self.params["box_fg_iou_thresh"] = 0.5	# 0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2
		self.params["box_bg_iou_thresh"] = 0.5	# 0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2

		self.params["trainable_backbone_layers"] = None #None, 5
		self.params["anchor_size"] = ((32,), (64,), (128,), (256,), (512,))
		self.params["anchor_aspect_ratio"] = ((0.5, 1.0, 2.0),) * len(self.params["anchor_size"])
		self.params["roipooler_outputSize"] = 7
		self.params["roipooler_sampling_ratio"] = 1
		self.params["roipooler_canonical_scale"] = 224
		self.params["roipooler_canonical_level"] = 4

		self.params["input_size_boxHead"] = 7
		self.params["conv_layers_boxHead"] = [256, 256, 256, 256]
		self.params["fc_layers_boxHead"] = [1024]

		self.params["aug_policy"] = {}
		self.params["aug_policy"]["mode"] = "oneOF"   # sequential, oneOF
		self.params["aug_policy"]["oneOF"] = [1.0, 0.0, 1.0]

		self.params["aug_policy"]['NPS_params'] = [(1,100), 0.0]
		self.params["aug_policy"]['HorizontalFlip'] = 1.0 #1.0
		self.params["aug_policy"]['VerticalFlip'] = 1.0 #1.0
		self.params["aug_policy"]["RandomRotate90"] = [['xy'], 0.0]
		self.params["aug_policy"]["Rotate"] = [['xy'], 180, 'constant', -1024, 0.0]
		self.params["aug_policy"]["SliceFlip"] = 0.0
		self.params["aug_policy"]["Transpose"] = 0.0
		#self.params["aug_policy"]["RandomCrop"] = [256, 256, 3, 1.0]
		#self.params["aug_policy"]["RandomSizedCrop"] = [(512,512,3), 1.0, 0.5, 1.0, 2, 1.0]
		self.params["aug_policy"]["ShiftScaleRotate"] = [0.0625, 0.2, 45, 'mirror', 0.0]

		self.params["aug_policy"]["Blur"] = [[11,13], True, 'constant', -1024, 0.0]
		self.params["aug_policy"]["Downscale"] = [0.2, 0.2, 1.0]
		self.params["aug_policy"]["Sharpen"] = [(0.01,0.02), (0.5,1), 'constant', -1024, 0.0]
		self.params["aug_policy"]["RandomBrightnessContrast"] = [4000, 0.2, 0.9, 0.0]
		self.params["aug_policy"]["RandomGamma"] = [(-10,10), 0.1]
		#self.params["aug_policy"]["UnsharpMask"] = [(13, 15), 0, 1.0, 'constant', 1.0] doesn't work properly

		self.params["data_dir"] = "/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/numpy_dataset/3_consecutive_slice"
		self.params["gt_annotations_dir"] = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/Documents_Information/DL_info.csv'
		self.params["saved_models_dir"] = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/Models/resnet/without_dicuagmentandwithouttransform'
		self.params["unfreeze"] = True

