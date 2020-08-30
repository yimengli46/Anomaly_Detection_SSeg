import os

class Parameters(object):
	def __init__(self):
		self.backbone = 'resnet' #'resnet', 'xception', 'drn', 'mobilenet'
		self.out_stride = 16 #8
		self.dataset = 'cityscapes' # 'pascal', 'coco', 'cityscapes'
		self.checkname = 'deeplab_duq'
		self.use_sbd = True
		self.workers = 4
		self.base_size = 1024
		self.crop_size = 768
		self.resize_ratio = 1.0
		self.sync_bn = False
		self.freeze_bn = False
		self.loss_type = 'bce' # 'ce', 'focal'

		# training hyper params
		self.epochs = 200
		self.batch_size = 2
		self.test_batch_size = self.batch_size
		self.use_balanced_weights = False

		# optimizer params
		self.lr = 0.1
		self.lr_scheduler = 'poly' # 'poly', 'step', 'cos'

		# cuda, seed and logging
		self.cuda = True
		self.gpu_id = '1'

		# checking point
		self.resume = None

		# finetuning pre-trained models
		self.ft = False

		# evaluation option
		self.eval_interval = 2
		self.no_val = False

		# duq params
		self.duq_centroid_size = 256
		self.duq_model_output_size = 64
		self.duq_learning_rate = 0.05
		self.duq_l_gradient_penalty = 0.5
		self.duq_gamma = 0.999
		self.duq_length_scale = 0.1
		self.duq_weight_decay = 5e-4
