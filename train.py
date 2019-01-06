#!/usr/bin/env python
import argparse
from reverse_matchmove import *


parser = argparse.ArgumentParser()


parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--dataset', nargs='?', default='chiang_mai', type=str)
parser.add_argument('--batch_size', nargs='?', default=5, type=int)
parser.add_argument('--workers', nargs='?', default=8, type=int)
parser.add_argument('--res', nargs='?', default=512, type=int)
parser.add_argument('--lr_drop_every', nargs='?', default=50, type=int)
parser.add_argument('--lr_drop_start', nargs='?', default=50, type=int)
parser.add_argument('--vgg_layers_p', type=int, nargs='+', default=[2, 4, 6, 8])
parser.add_argument('--vgg_layers_p_weight', type=int, nargs='+', default=[1., 1., 1., 1.])
parser.add_argument('--l1_weight', nargs='?', default=3., type=float)
parser.add_argument('--dp_mult', nargs='?', default=.5, type=float)
parser.add_argument('--disc_mult', nargs='?', default=.5, type=float)
parser.add_argument('--disc_start', nargs='?', default=100, type=int)
parser.add_argument('--perceptual_weight', nargs='?', default=2.5, type=float)
parser.add_argument('--vgg_weight_div', nargs='?', default=1, type=float)
parser.add_argument('--lr', nargs='?', default=2e-4, type=float)
parser.add_argument('--train_epoch', nargs='?', default=200, type=int)
parser.add_argument('--weight_decay', nargs='?', default=.000001, type=float)
parser.add_argument('--beta1', nargs='?', default=.5, type=float)
parser.add_argument('--beta2', nargs='?', default=.999, type=float)
parser.add_argument('--drop', nargs='?', default=.01, type=float)
parser.add_argument('--center_drop', nargs='?', default=.01, type=float)
parser.add_argument('--save_every', nargs='?', default=5, type=int)
parser.add_argument('--save_img_every', nargs='?', default=1, type=int)
parser.add_argument('--ids_test', type=int, nargs='+', default=[0, 100])
parser.add_argument('--ids_train', type=int, nargs='+', default=[0, 2])
parser.add_argument('--save_root', nargs='?', default='chiang_mai', type=str)
parser.add_argument('--load_state', nargs='?', type=str)
parser.add_argument('--reset', nargs='?', default=0, type=int)


params = vars(parser.parse_args())
print(params['vgg_layers_p_weight'])
# if load_state arg is not used, then train model from scratch
if __name__ == '__main__':
    sr = ReverseMatchmove(params)
    if params['load_state']:
        if params['reset'] == 1:
            sr.load_state(params['load_state'],reset=True)
        else:
            sr.load_state(params['load_state'],reset=False)
    else:
        print('Starting From Scratch')
    sr.train()

revtrack = ReverseMatchmove(params)
revtrack.train()
