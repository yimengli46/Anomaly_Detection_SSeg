import numpy as np
import cv2
import json
import matplotlib.pyplot as plt 
from PIL import Image 
import os
from utils import apply_color_map

#==================================================set up parameters===================================================================
dataset_base_folder = '/projects/kosecka/yimeng/Datasets/{}'.format('Fishyscapes_Static')
result_folder = 'results_SN/resNet_Fishyscapes' #'results/resNet_roadAnomaly', 'results/mobileNet_roadAnomaly'

#===================================================== process output ===============================================================
for i in range(30):
	print('i = {}'.format(i))

	np_output = np.load('{}/{}_result.npy'.format(result_folder, i), allow_pickle=True).item()
	class_prediction = np_output['sseg']
	uncertainty_mat = np_output['uncertainty']

	rgb_img = cv2.imread('{}/{}.png'.format(dataset_base_folder, i))[:,:,::-1]
	label_img = cv2.imread('{}/{}_label.png'.format(dataset_base_folder, i), 0)

	# in case the result is got from downsampled images
	h, w = uncertainty_mat.shape
	label_img = cv2.resize(label_img, (w, h), interpolation=cv2.INTER_NEAREST)
	
	# ignored pixel has label 2
	mask_ignored = (label_img == 2)
	uncertainty_mat[mask_ignored] = 1.0
	# outlier have label 1, others have label 0
	#label_img = np.where(label_img != 1, 0, label_img)

	uncertainty_mat = 1 - uncertainty_mat

	colored_prediction_array = apply_color_map(class_prediction)

	# visualization
	fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
	ax[0][0].imshow(rgb_img)
	ax[0][0].get_xaxis().set_visible(False)
	ax[0][0].get_yaxis().set_visible(False)
	ax[0][0].set_title('RGB Image')
	ax[0][1].imshow(label_img)
	ax[0][1].get_xaxis().set_visible(False)
	ax[0][1].get_yaxis().set_visible(False)
	ax[0][1].set_title('Label Image')
	ax[1][0].imshow(colored_prediction_array)
	ax[1][0].get_xaxis().set_visible(False)
	ax[1][0].get_yaxis().set_visible(False)
	ax[1][0].set_title('Semantic Segmentation')
	ax[1][1].imshow(uncertainty_mat, vmin=0.0, vmax=1.0)
	ax[1][1].get_xaxis().set_visible(False)
	ax[1][1].get_yaxis().set_visible(False)
	ax[1][1].set_title('Uncertainty')

	fig.tight_layout()
	fig.savefig('{}/{}_vis.jpg'.format(result_folder, i))
	plt.close()





