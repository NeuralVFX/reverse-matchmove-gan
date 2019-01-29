import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils import spectral_norm


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


def gen_conv_block(ni, nf, kernel_size=3, icnr=True, drop=.1):
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


def disc_con_block(ni, nf, kernel_size=3, stride=1):
    # conv_block with spectral normalization

    layers = []
    conv = spectral_norm(nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2, stride=stride))
    relu = nn.LeakyReLU(inplace=True)

    layers += [conv, relu]
    return nn.Sequential(*layers)


class UpResBlock(nn.Module):
    # Upres block which uses pixel shuffle with res connection
    def __init__(self, ic, oc, kernel_size=3, drop=.1):
        super(UpResBlock, self).__init__()
        self.oc = oc
        self.conv = gen_conv_block(ic, oc * 4, kernel_size=kernel_size, drop=drop)
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
        operations += [nn.ConvTranspose2d(in_channels=ic,
                                                        out_channels=oc,
                                                        padding=padding,
                                                        output_padding=0,
                                                        kernel_size=kernel_size,
                                                        stride=stride,
                                                        bias=False)]

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


class DownRes(nn.Module):
    # Add Layer of Spatia Mapping
    def __init__(self, ic, oc, kernel_size=3):
        super(DownRes, self).__init__()
        self.kernel_size = kernel_size
        self.oc = oc
        self.conv = disc_con_block(ic, oc, kernel_size=kernel_size, stride=2)

    def forward(self, x):
        unsqueeze_x = x.unsqueeze(0)

        x = self.conv(x)

        upres_x = nn.functional.interpolate(unsqueeze_x, size=[self.oc, x.shape[2], x.shape[3]], mode='trilinear',
                                            align_corners=True)[0]
        x = x + (upres_x * .2)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.query = spectral_norm(nn.Conv1d(in_channel, in_channel // 8, 1))
        self.key = spectral_norm(nn.Conv1d(in_channel, in_channel // 8, 1))
        self.value = spectral_norm(nn.Conv1d(in_channel, in_channel, 1))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input
        return out

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
            print ('up_block')
            operations += [UpResBlock(int(min(max_filts, filt_count * 2)), int(min(max_filts, filt_count)), drop=drop)]
            #if a == 2:
            #    print('attn')
            #    att =  SelfAttention(int(min(max_filts, filt_count * 2)))

                #operations += [SelfAttention(int(min(max_filts, filt_count * 2)))]
            filt_count = int(filt_count * 2)

        operations += [
            TransposeBlock(ic=filts, oc=int(min(max_filts, filt_count)), kernel_size=kernel_size, padding=1,
                           drop=center_drop),

            TransposeBlock(ic=z_size, oc=filts, kernel_size=kernel_size, padding=0, stride=1, drop=center_drop)
        ]

        operations.reverse()

        operations += [nn.ReflectionPad2d(3),
                       spectral_norm(nn.Conv2d(in_channels=min_filts, out_channels=channels, kernel_size=7, padding=0, stride=1))]

        self.model = nn.Sequential(*operations)
        #self.att = att
    #def fix_net(self):
    #    fix = list(self.model.children())[:5] + [self.att] + list(self.model.children())[5:]
    #    print (fix)
    #    self.model = nn.Sequential(*fix)
    def forward(self, x):
        x = self.model(x)
        return F.tanh(x)


class Discriminator(nn.Module):
    # Using reverse shuffling should reduce the repetitive shimmering patterns
    def __init__(self, channels=3, filts_min=128, filts=512, use_frac=False, kernel_size=4, frac=None, layers=3):
        super(Discriminator, self).__init__()
        self.use_frac = use_frac
        operations = []
        if use_frac:
            self.frac = frac

        in_operations = [nn.ReflectionPad2d(3),
                         spectral_norm(
                             nn.Conv2d(in_channels=channels, out_channels=filts_min, kernel_size=7, stride=1))]

        filt_count = filts_min

        for a in range(layers):
            operations += [DownRes(ic=min(filt_count, filts), oc=min(filt_count * 2, filts), kernel_size=3)]
            #if a == 0:
            #    operations += [SelfAttention(min(filt_count * 2, filts))]
            print(min(filt_count * 2, filts))
            filt_count = int(filt_count * 2)

        out_operations = [
                spectral_norm(
                nn.Conv2d(in_channels=min(filt_count, filts), out_channels=1, padding=1, kernel_size=kernel_size,
                          stride=1))]

        operations = in_operations + operations + out_operations
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        # Run operations, and return relu activations for loss function
        if self.use_frac:
            x = self.frac(x)

        x = self.operations(x)
        return x


def make_vgg(depth=9):
    # VGG with max pool removed, cut only to the depth we wil use
    vgg = models.vgg19(pretrained=True)
    children = list(vgg.children())
    children.pop()

    # remove max pool to reduce JPG looking artifacts
    del children[0][4]
    del children[0][8]
    operations = children[0][:depth]

    for op in operations:
        op.no_init = True

    vgg = nn.Sequential(*operations)

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

    def __init__(self, vgg, ct_wgt, l1_weight, perceptual_layer_ids, weight_list, hooks=None):
        super().__init__()
        self.m, self.ct_wgt, self.l1_weight = vgg, ct_wgt, l1_weight

        if not hooks:
            self.cfs = [SetHook(vgg[i]) for i in perceptual_layer_ids]
        else:
            print('Using custom hooks')
            self.cfs = hooks

        ratio = ct_wgt / sum(weight_list)
        weight_list = [a * ratio for a in weight_list]
        self.weight_list = weight_list

    def forward(self, fake_img, real_img, disc_mode=False):
        # Calculate L1 and Perceptual Loss
        if sum(self.weight_list) > 0.0:
            self.m(real_img.data)
            targ_feats = [o.feats.data.clone() for o in self.cfs]
            fake_result = self.m(fake_img)
            inp_feats = [o.feats for o in self.cfs]
            result_perc = [F.l1_loss(inp.view(-1), targ.view(-1)) * layer_weight for inp, targ, layer_weight in
                           zip(inp_feats, targ_feats, self.weight_list)]
        else:
            result_perc = [torch.zeros(1).cuda() for layer_weight in self.weight_list]
            fake_result = torch.zeros(1).cuda()

        if not disc_mode:
            result_l1 = [F.l1_loss(fake_img, real_img) * self.l1_weight]
            return result_perc, result_l1
        else:
            return result_perc, fake_result

    def close(self):
        [o.remove() for o in self.sfs]
