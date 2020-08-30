from PIL import Image
import numpy as np

def resize_targets_img(par, targets_copy):
    new_size = int(par.crop_size*0.25)

    N, _, _ = targets_copy.shape

    downsampled_targets = np.zeros((N, new_size, new_size))

    for i in range(N):
        temp_targets = Image.fromarray(targets_copy[i])
        temp_targets = temp_targets.resize((new_size, new_size), Image.NEAREST)
        downsampled_targets[i] = np.array(temp_targets)
        
    return downsampled_targets