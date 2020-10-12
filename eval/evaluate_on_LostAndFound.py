import numpy as np
from utils import compute_iou
import cv2

result_base_folder = 'results_duq/resNet_lostAndFound' #'results/resNet_lostAndFound_2' #'results/mobileNet_lostAndFound'
dataset_base_folder = '/projects/kosecka/yimeng/Datasets/{}'.format('Lost_and_Found')

#uncertainty_threshold_list = [x/10.0 for x in range(5, 50)]
uncertainty_threshold_list = [x/100.0 for x in range(5, 95, 5)]

#big_outlier_list = [2, 3, 4, 7, 10, 11, 15, 16, 25, 27, 31, 33, 34, 35, 38, 40, 45, 46, 48, 50, 51, 54, 57, 60, 61, 63, 65, 68, 71, 72, 74, 76, 83, 84, 85, 86, 91, 93, 95]

#big_outlier_list = [2, 4, 10, 16, 27, 38, 45, 50, 51, 61, 65, 68, 74, 76, 83, 84, 95]

big_outlier_list = [2, 4, 16, 27, 45, 51, 61, 65, 68, 76, 83, 84]

num_test_imgs = len(big_outlier_list)
all_mIoU = np.zeros(num_test_imgs)

for uncertainty_threshold in uncertainty_threshold_list:
	for i, img_id in enumerate(big_outlier_list):
		#print('img_id = {}'.format(img_id))

		result = np.load('{}/{}_result.npy'.format(result_base_folder, img_id), allow_pickle=True).item()
		uncertainty_result = result['uncertainty']
		
		label_img = cv2.imread('{}/{}_label.png'.format(dataset_base_folder, img_id), 0)
		# in case the result is got from downsampled images
		h, w = uncertainty_result.shape
		label_img = cv2.resize(label_img, (w, h), interpolation=cv2.INTER_NEAREST)
		
		# ignored pixel has label 2
		mask_ignored = (label_img == 2)
		uncertainty_result[mask_ignored] = 1.0
		# outlier have label 1, others have label 0
		label_img = np.where(label_img != 1, 0, label_img)

		uncertainty_result = np.where(uncertainty_result < uncertainty_threshold, 1, 0)		

		mIoU, all_IoU, idx_not_zero = compute_iou(label_img, uncertainty_result, 2)
		# only take the IoU on the outlier, rather than the background
		#print('all_IoU = {}'.format(all_IoU))
		all_mIoU[i] = all_IoU[1]

	print('uncertainty_threshold = {}'.format(uncertainty_threshold))
	print('mean IoU over {} imgs from LostAndFound is {:.4f}'.format(num_test_imgs, np.mean(all_mIoU)))