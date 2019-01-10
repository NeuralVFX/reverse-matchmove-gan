import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils import spectral_norm


############################################################################
# Re-usable blocks
############################################################################


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

class PadClip(nn.Module):
    # Upres block which uses pixel shuffle with res connection

    def __init__(self, padding = 1):
        super(PadClip, self).__init__()
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
    def forward(self,x):
        x = self.pad(x)
        if self.padding % 2 == 0:
            x = x[:,:,:-1,:-1]
        return x



def spectral_conv_block(ni, nf, kernel_size=3):
    # conv_block with spectral normalization
    layers = []
    padding = PadClip(padding = kernel_size//2)
    conv = spectral_norm(nn.Conv2d(ni, nf, kernel_size))
    relu = nn.LeakyReLU(inplace=True)

    layers += [padding, conv, relu]
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
        operations += [nn.ConvTranspose2d(in_channels=ic, out_channels=oc, padding=padding, output_padding=0,
                                         kernel_size=kernel_size, stride=stride, bias=False)]

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


class ReverseShuffle(nn.Module):
    # Pixel Shuffle which replaces conv stride 2
    def __init__(self):
        super(ReverseShuffle, self).__init__()

    def forward(self, fm):
        r = 2
        b, c, h, w = fm.shape
        out_channel = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        fm_view = fm.contiguous().view(b, c, out_h, r, out_w, r)
        fm_prime = fm_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)
        return fm_prime


class DownRes(nn.Module):
    # Add Layer of Spatia Mapping
    def __init__(self, ic, oc, kernel_size=3):
        super(DownRes, self).__init__()
        self.kernel_size = kernel_size
        self.oc = oc
        self.conv = spectral_conv_block(ic, oc // 4, kernel_size=kernel_size)
        self.rev_shuff = ReverseShuffle()

    def forward(self, x):
        unsqueeze_x = x.unsqueeze(0)

        x = self.conv(x)
        x = self.rev_shuff(x)

        upres_x = nn.functional.interpolate(unsqueeze_x, size=[self.oc, x.shape[2], x.shape[3]], mode='trilinear',
                                            align_corners=True)[0]
        x = x + (upres_x * .2)
        return x


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

        operations += [
            TransposeBlock(ic=filts, oc=int(min(max_filts, filt_count)), kernel_size=kernel_size, padding=1,
                           drop=center_drop),

            TransposeBlock(ic=z_size, oc=filts, kernel_size=kernel_size, padding=0, stride=1, drop=center_drop)
        ]

        operations.reverse()

        operations += [nn.ReflectionPad2d(3),
                       nn.Conv2d(in_channels=min_filts, out_channels=channels, kernel_size=7, padding=0, stride=1)]

        self.model = nn.Sequential(*operations)

    def forward(self, x):
        x = self.model(x)
        return F.tanh(x)


class UnshuffleDiscriminator(nn.Module):
    # Using reverse shuffling should reduce the repetitive shimmering patterns
    def __init__(self, channels=3, filts_min=128, filts=128, use_frac=False, kernel_size=4, frac=None, layers=3,
                 drop=.01):
        super(UnshuffleDiscriminator, self).__init__()
        self.use_frac = use_frac
        operations = []
        if use_frac:
            self.frac = frac

        in_operations = [nn.ReflectionPad2d(3),
                         spectral_norm(nn.Conv2d(in_channels=channels, out_channels=filts_min, kernel_size=7, stride=1))]

        filt_count = filts_min

        for a in range(layers):
            operations += [DownRes(ic=min(filt_count, filts), oc=min(filt_count * 2, filts), kernel_size=4)]
            print(min(filt_count * 2, filts))
            filt_count = int(filt_count * 2)

        out_operations = [
            spectral_norm(nn.Conv2d(in_channels=min(filt_count, filts), out_channels=1, padding=1, kernel_size=kernel_size,
                      stride=1))]

        operations = in_operations + operations + out_operations
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        # Run operations, and return relu activations for loss function
        if self.use_frac:
            x = self.frac(x)

        x = self.operations(x)
        return x



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

    def __init__(self, vgg, ct_wgt, l1_weight, perceptual_layer_ids, weight_list, hooks = None):
        super().__init__()
        self.m, self.ct_wgt, self.l1_weight = vgg, ct_wgt, l1_weight

        if not hooks:
            self.cfs = [SetHook(vgg[i]) for i in perceptual_layer_ids]
        else:
            print ('Using custom hooks')
            self.cfs = hooks

        ratio = ct_wgt / sum(weight_list)
        weight_list = [a * ratio for a in weight_list]
        self.weight_list = weight_list

    def forward(self, fake_img, real_img):
        # Calculate L1 and Perceptual Loss
        self.m(real_img.data)
        targ_feats = [o.feats.data.clone() for o in self.cfs]
        fake_result = self.m(fake_img)
        inp_feats = [o.feats for o in self.cfs]
        result_perc = [F.l1_loss(inp.contiguous().view(-1), targ.contiguous().view(-1)) * layer_weight for inp, targ, layer_weight in
                     zip(inp_feats, targ_feats, self.weight_list)]

        result_l1 = [F.l1_loss(fake_img, real_img) * self.l1_weight]

        return result_perc, result_l1, fake_result

    def close(self):
        [o.remove() for o in self.sfs]
