import os

class Parameters(object):
	def __init__(self):
		self.backbone = 'resnet' #'resnet', 'xception', 'drn', 'mobilenet'
		self.out_stride = 16 #8
		self.dataset = 'cityscapes' # 'pascal', 'coco', 'cityscapes'
		self.checkname = 'try_sseg'
		self.use_sbd = True
		self.workers = 4
		self.base_size = (2048, 1024)
		self.crop_size = (1024, 512)
		self.sync_bn = False
		self.freeze_bn = False
		self.loss_type = 'ce' # 'ce', 'focal'

		# training hyper params
		self.epochs = 200
		self.batch_size = 12
		self.test_batch_size = 12
		self.use_balanced_weights = False

		# optimizer params
		self.lr = 0.01
		self.lr_scheduler = 'poly' # 'poly', 'step', 'cos'

		# cuda, seed and logging
		self.cuda = True
		self.gpu_id = '1'

		# checking point
		self.resume = None
		self.checkname = 'deeplab_{}'.format(self.backbone)

		# finetuning pre-trained models
		self.ft = False

		# evaluation option
		self.eval_interval = 2
		self.no_val = False
