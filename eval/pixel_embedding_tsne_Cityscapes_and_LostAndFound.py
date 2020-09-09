import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import cv2
from sklearn import manifold, datasets
import random

result_base_folder = 'results/resNet_features'
saved_folder = 'results/resNet_features_vis'

num_classes = 19

id_cityscapes = 1
id_lostAndFound = 45 #4, 27

random.seed(0)

for id_cityscapes in range(10):
	result_cityscapes = np.load('{}/{}_cityscapes.npy'.format(result_base_folder, id_cityscapes), allow_pickle=True).item()
	features = np.transpose(result_cityscapes['feature'], (1, 2, 0)).reshape((-1, 64))
	labels = result_cityscapes['label'].flatten()
	print('features.shape = {}, labels.shape = {}'.format(features.shape, labels.shape))
	features = features[labels < 255]
	labels = labels[labels < 255]
	print('remove unlabled pixels ...')
	print('features.shape = {}, labels.shape = {}'.format(features.shape, labels.shape))

	if id_cityscapes == 0:
		all_features = features
		all_labels = labels
	else:
		all_features = np.concatenate([all_features, features], axis=0)
		all_labels   = np.concatenate([all_labels, labels], axis=0)

# randomly pick 100 points for each class. So pick 1900 points from features,
for i in range(num_classes):
	current_features = all_features[all_labels==i]
	current_labels   = all_labels[all_labels==i]

	chosen_idx = random.sample([x for x in range(current_features.shape[0])], 200)

	if i == 0:
		features = current_features[chosen_idx]
		labels   = current_labels[chosen_idx]
	else:
		features = np.concatenate([features, current_features[chosen_idx]], axis=0)
		labels   = np.concatenate([labels, current_labels[chosen_idx]], axis=0)

all_features = features
all_labels = labels

print('************all_features.shape = {}, all_labels.shape = {}************'.format(all_features.shape, all_labels.shape))

for id_lostAndFound in [4, 27, 45, 50, 51, 61]:
	features = all_features
	labels = all_labels

	result_lostAndFound =  np.load('{}/{}_LostAndFound.npy'.format(result_base_folder, id_lostAndFound), allow_pickle=True).item()
	features_Outlier = np.transpose(result_lostAndFound['feature'], (1, 2, 0)).reshape((-1, 64))
	labels_Outlier = result_lostAndFound['label'].flatten()

	features_Outlier = features_Outlier[labels_Outlier==1]
	labels_Outlier = labels_Outlier[labels_Outlier==1] * 19
	assert features_Outlier.shape[0] == labels_Outlier.shape[0]

	chosen_idx = random.sample([x for x in range(features_Outlier.shape[0])], 100)
	features_Outlier = features_Outlier[chosen_idx]
	labels_Outlier   = labels_Outlier[chosen_idx]
	print('features_Outlier.shape = {}, labels_Outlier.shape = {}'.format(features_Outlier.shape, labels_Outlier.shape))

	features = np.concatenate([features, features_Outlier], axis=0)
	labels   = np.concatenate([labels, labels_Outlier], axis=0)

	#==================================================================================================================================
	thing_list = ['road', 'sidewalk', 'building', 'wall', 'fence', \
		'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
		'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
		'motorcycle', 'bicycle', 'outlier', '--']

	thing_map = {}
	for idx, name in enumerate(thing_list):
		thing_map[name] = idx

	# visualize the bbox features through t-sne
	X = features
	y = labels.astype(np.int16)
	N = len(thing_list)

	tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, verbose=1)
	X_tsne = tsne.fit_transform(X)

	print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

	x_min, x_max = X_tsne.min(0), X_tsne.max(0)
	X_norm = (X_tsne - x_min) / (x_max - x_min)

	# define the colormap
	cmap = plt.cm.gist_ncar
	# extract all colors from the .jet map
	cmaplist = [cmap(i) for i in range(cmap.N)]
	# create the new map
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	# define the bins and normalize
	bounds = np.linspace(0,N-1,N)
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

	plt.style.use('dark_background')
	fig = plt.figure(figsize=(10, 10))
	scat = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap=cmap, norm=norm)
	cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
	cb.ax.set_yticklabels(thing_list)

	plt.xticks([])
	plt.yticks([])
	plt.title('number of points = {}, number of outlier = {}'.format(X.shape[0], features_Outlier.shape[0]))
	#plt.show()
	fig.tight_layout()
	fig.savefig('{}/lostAndFound_{}_cityscapes_all.jpg'.format(saved_folder, id_lostAndFound))
	plt.close()



#assert 1==2

