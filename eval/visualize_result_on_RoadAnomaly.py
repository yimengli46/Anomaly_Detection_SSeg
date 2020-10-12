import numpy as np
import cv2
import json
import matplotlib.pyplot as plt 
from PIL import Image 
import os
from utils import apply_color_map

#==================================================set up parameters===================================================================
dataset_base_folder = '/projects/kosecka/yimeng/Datasets/{}'.format('RoadAnomaly')
result_folder = 'results_duq_temp/resNet_roadAnomaly' #'results/resNet_roadAnomaly', 'results/mobileNet_roadAnomaly'

#===================================================== process output ===============================================================
for i in range(60):
	print('i = {}'.format(i))
	rgb_img = cv2.imread('{}/{}.png'.format(dataset_base_folder, i))[:,:,::-1]
	label_img = cv2.imread('{}/{}_label.png'.format(dataset_base_folder, i), 0)

	np_output = np.load('{}/{}_result.npy'.format(result_folder, i), allow_pickle=True).item()
	
	class_prediction = np_output['sseg']
	uncertainty_mat = np_output['uncertainty']

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





