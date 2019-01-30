import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import *
from torch.autograd import Variable
from matplotlib import animation
import matplotlib

matplotlib.use('agg')


############################################################################
# Helper Utilities
############################################################################


def icnr(x, scale=2, init=nn.init.orthogonal_):
    # initiate shuffle conv layers with ICNR

    new_shape = [int(x.shape[0]) // int(scale ** 2)] + list(x.shape[1:])
    single_kernel = torch.zeros(new_shape)
    single_kernel = init(single_kernel)
    single_kernel = single_kernel.transpose(0, 1)
    single_kernel = single_kernel.contiguous().view(single_kernel.shape[0], single_kernel.shape[1], -1)
    full_kernel = single_kernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    full_kernel = full_kernel.contiguous().view(transposed_shape)
    full_kernel = full_kernel.transpose(0, 1)
    return full_kernel

def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.ConvTranspose2d,)):
            return nn.utils.spectral_norm(m)
        else:
            return m




def weights_init_normal(m):
    # Set initial state of weights

    classname = m.__class__.__name__
    if hasattr(m, 'no_init'):
        print(f'Skipping Init on Pre-trained:{classname}')
    else:
        if 'ConvTrans' == classname:
            pass
        elif 'Linear' in classname:
            nn.init.normal(m.weight.data, 0, .02)
        elif 'Conv2d' in classname or 'ConvTrans' in classname:
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
            # only use ICNR if icrn attr found
            if classname == 'Conv2d' and hasattr(m, 'icnr'):
                kern = icnr(m.weight)
                m.weight.data.copy_(kern)
                print(f'Init with ICNR:{classname}')


def mft(tensor):
    # Return mean float tensor #
    return torch.mean(torch.FloatTensor(tensor))


############################################################################
# Display Images
############################################################################


def show_test(params, denorm, mtran, train_data, test_data, model, save=False):
    # Show and save

    ids_a = params['ids_test']
    ids_b = params['ids_train']
    image_grid_len = len(ids_a + ids_b)
    fig, ax = plt.subplots(image_grid_len, 2, figsize=(10, 4 * image_grid_len))
    count = 0
    model.eval()

    dataloader_train = torch.utils.data.DataLoader(train_data,
                                                   batch_size=1,
                                                   num_workers=params["workers"],
                                                   shuffle=True,
                                                   drop_last=True)
    dataloader_test = torch.utils.data.DataLoader(test_data,
                                                  batch_size=1,
                                                  num_workers=params["workers"],
                                                  shuffle=False,
                                                  drop_last=True)

    for idx, data in enumerate(dataloader_test):
        if idx in ids_a:
            real = Variable(data[0]).cuda()
            mat = Variable(data[1]).cuda()
            mat = mtran(mat)
            test = model(mat)
            ax[count, 0].cla()
            ax[count, 0].imshow(denorm.denorm(real[0]))
            ax[count, 1].cla()
            ax[count, 1].imshow(denorm.denorm(test[0]))
            count += 1
        if idx > max(ids_a)+1:
            break

    for idx, data in enumerate(dataloader_train):
        if idx in ids_b:
            real = Variable(data[0]).cuda()
            mat = Variable(data[1]).cuda()
            mat = mtran(mat)
            test = model(mat)
            ax[count, 0].cla()
            ax[count, 0].imshow(denorm.denorm(real[0]))
            ax[count, 1].cla()
            ax[count, 1].imshow(denorm.denorm(test[0]))

            count += 1
        if idx > max(ids_b)+1:
            break
    model.train()
    if save:
        plt.savefig(save)
    plt.show()
    plt.close(fig)


############################################################################
# Display Animations
############################################################################


def plot_movie_mp4(image_array, filename):
    # Take numpy image sequence and save as gif on disk

    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, interval=15, frames=len(image_array))
    anim.save(filename, writer='imagemagick')


def test_repo(mdl, dataset, filename):
    # Use "REPO" dataset to render out sequence and save as a gif

    dataloader_test = torch.utils.data.DataLoader(dataset,
                                                  batch_size=1,
                                                  num_workers=mdl.params["workers"],
                                                  shuffle=False, drop_last=True)
    denorm = mdl.transform
    anim_test = []
    count = 0
    mdl.model_dict['G'].eval()
    for (real, mat) in dataloader_test:
        mat = mdl.mtran(Variable(mat.cuda()))
        test = mdl.model_dict['G'](mat)
        for i in range(test.shape[0]):
            anim_test.append(denorm.denorm(test[0]))
            count += 1

    plot_movie_mp4(anim_test, filename)
    mdl.test_data.repo = False
    mdl.model_dict['G'].train()
