import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


############################################################################
# Re-usable blocks
############################################################################


class TensorTransform(nn.Module):
    # Used to convert between default color space and VGG colorspace

    def __init__(self, res=256, mean=[.485, .456, .406], std=[.229, .224, .225]):
        super(TensorTransform, self).__init__()

        self.mean = torch.zeros([3, res, res]).cuda()
        self.mean[0, :, :] = mean[0]
        self.mean[1, :, :] = mean[1]
        self.mean[2, :, :] = mean[2]

        self.std = torch.zeros([3, res, res]).cuda()
        self.std[0, :, :] = std[0]
        self.std[1, :, :] = std[1]
        self.std[2, :, :] = std[2]

    def forward(self, x):
        norm_ready = (x * .5) + .5
        result = (norm_ready - self.mean) / self.std
        return result


class MatrixTransform(nn.Module):
    # Used to scale input by mean and standard deviation

    def __init__(self, mean, std):
        super(MatrixTransform, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        result = (x - self.mean) / self.std
        return result


def conv_block(ni, nf, kernel_size=3, icnr=True, drop=.1):
    # Conv block which stores ICNR attribute for initialization
    layers = []
    conv = nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2)
    if icnr:
        conv.icnr = True

    relu = nn.LeakyReLU(inplace=True)

    bn = nn.BatchNorm2d(nf)
    drop = nn.Dropout(drop)
    layers += [conv, relu, bn, drop]
    return nn.Sequential(*layers)


class UpResBlock(nn.Module):
    # Upres block which uses pixel shuffle with res connection

    def __init__(self, ic, oc, kernel_size=3, drop=.1):
        super(UpResBlock, self).__init__()
        self.oc = oc
        self.conv = conv_block(ic, oc * 4, kernel_size=kernel_size, drop=drop)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        # store input for res
        unsqueeze_x = x.unsqueeze(0)

        x = self.conv(x)
        x = self.ps(x)
        # resize input with interpolations and add as res connection
        upres_x = nn.functional.interpolate(unsqueeze_x,
                                            size=[self.oc, x.shape[2], x.shape[3]],
                                            mode='trilinear',
                                            align_corners=True)[0]
        return x + (upres_x * .2)


class TransposeBlock(nn.Module):
    # Transpose Convolution with res connection

    def __init__(self, ic=4, oc=4, kernel_size=3, padding=1, stride=2, drop=.001):
        super(TransposeBlock, self).__init__()
        self.oc = oc
        if padding is None:
            padding = int(kernel_size // 2 // stride)

        operations = []
        operations += nn.ConvTranspose2d(in_channels=ic, out_channels=oc, padding=padding, output_padding=0,
                                         kernel_size=kernel_size, stride=stride, bias=False)

        operations += [nn.LeakyReLU(inplace=True), nn.BatchNorm2d(oc), nn.Dropout(drop)]

        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        # store input
        unsqueeze_x = x.unsqueeze(0)

        # run block
        x = self.operations(x)

        # resize input with interpolations and add as res connection
        res_x = nn.functional.interpolate(unsqueeze_x,
                                          size=[self.oc, x.shape[2], x.shape[3]],
                                          mode='trilinear',
                                          align_corners=True)[0]
        return x + (res_x * .2)


############################################################################
# Generator and VGG
############################################################################


class Generator(nn.Module):
    # Generator to convert Z sized vector to image

    def __init__(self, layers=6, z_size=13, filts=1024, max_filts=512, min_filts=128, kernel_size=4, channels=3,
                 drop=.1, center_drop=.1):
        super(Generator, self).__init__()
        operations = []

        filt_count = min_filts

        for a in range(layers):
            operations += [UpResBlock(int(min(max_filts, filt_count * 2)), int(min(max_filts, filt_count)), drop=drop)]
            filt_count = int(filt_count * 2)
            print(filt_count)

        operations += [
            TransposeBlock(ic=filts, oc=int(min(max_filts, filt_count)), kernel_size=kernel_size, padding=1,
                           drop=center_drop),

            TransposeBlock(ic=z_size, oc=filts, kernel_size=kernel_size, padding=0, stride=1, drop=center_drop)
        ]

        operations.reverse()

        operations += [nn.ReflectionPad2d(3),
                       nn.Conv2d(in_channels=min_filts + 2, out_channels=channels, kernel_size=7, padding=0, stride=1)]

        self.model = nn.Sequential(*operations)

    def forward(self, x):
        x = self.model(x)
        return F.tanh(x)


def make_vgg():
    vgg = models.vgg19(pretrained=True)
    children = list(vgg.children())
    children.pop()
    vgg = nn.Sequential(*children[0][:15])
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


############################################################################
# Hook and Losses
############################################################################


class SetHook:
    # Register hook inside of network to retrieve features
    feats = None

    def __init__(self, block):
        self.hook_reg = block.register_forward_hook(self.hook)

    def hook(self, module, hook_input, output):
        self.feats = output

    def close(self):
        self.hook_reg.remove()


class PerceptualLoss(nn.Module):
    # Store Hook, Calculate Perceptual Loss

    def __init__(self, vgg, ct_wgt, l1_weight, perceptual_layer_ids, weight_div=1):
        super().__init__()
        self.m, self.ct_wgt, self.l1_weight = vgg, ct_wgt, l1_weight
        self.cfs = [SetHook(vgg[i]) for i in perceptual_layer_ids]

        # make weight list, tapered by weight div
        weight = 1.0
        weight_list = []
        for i in range(len(self.cfs)):
            weight_list.append(weight)
            weight /= weight_div

        ratio = ct_wgt / sum(weight_list)
        weight_list = [a * ratio for a in weight_list]
        weight_list.reverse()

        self.weight_list = weight_list

    def forward(self, input_img, target_img):
        # Calculate L1 and Perceptual Loss
        self.m(target_img.data)
        result_l1 = [F.l1_loss(input_img, target_img) * self.l1_weight]
        targ_feats = [o.feats.data.clone() for o in self.cfs]
        self.m(input_img)
        inp_feats = [o.feats for o in self.cfs]
        result_ct = [F.l1_loss(inp.view(-1), targ.view(-1)) * layer_weight for inp, targ, layer_weight in
                     zip(inp_feats, targ_feats, self.weight_list)]
        return result_ct, result_l1

    def close(self):
        [o.remove() for o in self.sfs]
