import os

class Parameters(object):
	def __init__(self):
		backbone = 'resnet' #'resnet', 'xception', 'drn', 'mobilenet'
		out_stride = 16 #8
		dataset = 'cityscapes' # 'pascal', 'coco', 'cityscapes'
		use_sbd = True
		workers = 4
		base_size = 513
		crop_size = 513
		sync_bn = False
		freeze_bn = False
		loss_type = 'ce' # 'ce', 'focal'

		# training hyper params
		epochs = 200
		batch_size = 3
		test_batch_size = 3
		use_balanced_weights = False

		# optimizer params
		lr = 0.01
		lr_scheduler = 'poly' # 'poly', 'step', 'cos'
		momentum = 0.9
		weight_decay = 5e-4
		nesterov = False

		# cuda, seed and logging
		gpu_id = '1'

		# checking point
		resume = None
		checkname = 'deeplab_{}'.format(backbone)

		# finetuning pre-trained models
		ft = False

		# evaluation option
		eval_interval = 5
		no_val = False
