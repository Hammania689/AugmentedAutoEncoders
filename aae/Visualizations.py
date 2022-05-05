from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_img(img):
    fig = plt.figure(figsize=(10, 10), tight_layout=True, )    
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.imshow(img)

def plot_batch(imgs : List[np.ndarray]):
    batch_size = len(imgs)
    rows = batch_size // 4
    cols = 4
    
    fig = plt.figure(figsize=(15, 15),)

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     share_all=True)

    for ax, im in zip(grid, imgs):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im) 

    plt.show()
