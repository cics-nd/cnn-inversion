import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.switch_backend('agg')

# TODO: visdom

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# plotting
def plot_loss(save_dir, logger):
    fig = plt.figure()
    steps = len(logger['loss_d'])
    plt.plot(range(1, steps + 1), logger['loss_d'], label='loss_d')
    plt.plot(range(1, steps + 1), logger['loss_g'], label='loss_g')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig(save_dir + '/loss.pdf', dpi=300)
    plt.close(fig)

# plot images in grid
def plot_grid(save_dir, images, epoch, name, nrow=4):
    """
    Args:
        images (Tensor): B x C x H x W
    """
    if images.is_cuda:
        images = images.cpu()
    images = images.numpy()

    if images.shape[0] < 10:
        nrow = 2
        ncol = images.shape[0] // nrow
    else:
        ncol = nrow
    for c in range(images.shape[1]):

        fig = plt.figure(1, (11, 12))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(nrow, ncol),
                         axes_pad=0.1,
                         share_all=False,
                         cbar_location="top",
                         cbar_mode="single",
                         cbar_size="3%",
                         cbar_pad=0.1
                         )
        for j, ax in enumerate(grid):
            im = ax.imshow(images[j][c], cmap='jet', origin='lower')
            ax.set_axis_off()
            ax.set_aspect('equal')
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.toggle_label(True)
        plt.savefig(save_dir + '/{}_c{}_epoch{}.png'.format(name, c, epoch),
                    bbox_inches='tight')
        plt.close(fig)
