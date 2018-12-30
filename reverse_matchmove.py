import math
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from util import helpers as helper
from util import loaders as load
from models import networks as n

plt.switch_backend('agg')


############################################################################
# Train
############################################################################


class ReverseMatchmove:
    """
    Example usage if not using command line:

    from reverse_track import *

    params = {'dataset': 'chiang_mai',
              'batch_size': 5,
              'workers': 8,
              'res': 512,
              'drop': 0.01,
              'center_drop': .01,
              'weight_decay': .000001,
              'vgg_layers_c': [2,7,12],
              'vgg_weight_div': 1,
              'lr': 2e-4,
              'disc_layers': 3,
              'beta1': .5,
              'beta2': .999,
              'content_weight': 2.5,
              'l1_weight': 3.,
              'train_epoch': 200,
              'save_every': 5,
              'ids_test': [0, 100],
              'ids_train': [0, 2],
              'save_img_every': 1,
              'lr_drop_start': 0,
              'lr_drop_every': 5,
              'save_root': 'vgg_up'}

    rev = ReverseMatchmove(params)
    rev.train()

    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0

        # Setup data loaders
        self.transform = load.NormDenorm([.5, .5, .5], [.5, .5, .5])

        self.train_data = load.ImageMatrixDataset(f'./data/{params["dataset"]}/dataset_train.csv', self.transform,
                                                  output_res=params["res"], train=True)

        self.test_data = load.ImageMatrixDataset(f'./data/{params["dataset"]}/dataset_test.csv', self.transform,
                                                 output_res=params["res"], train=False)

        self.repo_data = load.ImageMatrixDataset(f'./data/{params["dataset"]}/dataset_repo.csv', self.transform,
                                                 output_res=params["res"], train=False, repo=True)

        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=params["batch_size"],
                                                        num_workers=params["workers"],
                                                        shuffle=True,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_data,
                                                       batch_size=params["batch_size"],
                                                       num_workers=params["workers"],
                                                       shuffle=False,
                                                       drop_last=True)

        self.data_len = self.train_data.__len__()
        print(f'Data Loaders Initialized,  Data Len:{self.train_data.__len__()}')

        # Setup models
        self.vgg_tran = n.TensorTransform(res=params["res"], mean=[.485, .456, .406], std=[.229, .224, .225])
        self.vgg_tran.cuda()

        self.model_dict['G'] = n.Generator(drop=params['drop'], center_drop=params['center_drop'])

        self.vgg = n.make_vgg()
        self.vgg.cuda()

        for i in self.model_dict.keys():
            self.model_dict[i].apply(helper.weights_init_normal)
            self.model_dict[i].cuda()
            self.model_dict[i].train()

        print('Networks Initialized')

        # Setup loss
        self.ct_loss = n.PerceptualLoss(self.vgg,
                                        params['content_weight'],
                                        params['l1_weight'],
                                        params['vgg_layers_c'],
                                        weight_div=params['vgg_weight_div'])

        self.ct_loss.cuda()

        # Setup optimizers
        self.opt_dict["G"] = optim.Adam(self.model_dict["G"].parameters(),
                                        lr=params['lr'],
                                        betas=(params['beta1'],
                                               params['beta2']),
                                        weight_decay=params['weight_decay'])

        print('Losses Initialized')

        # Setup history storage
        self.losses = ['L1_Loss', 'C_Loss']
        self.loss_batch_dict = {}
        self.loss_batch_dict_test = {}
        self.loss_epoch_dict = {}
        self.loss_epoch_dict_test = {}
        self.train_hist_dict = {}
        self.train_hist_dict_test = {}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []
            self.train_hist_dict_test[loss] = []
            self.loss_epoch_dict_test[loss] = []
            self.loss_batch_dict_test[loss] = []

        # measure stats of input data, create transform for this
        mat_mean, mat_std = self.measure_data()
        self.mtran = n.MatrixTransform(mean=torch.FloatTensor(mat_mean).cuda(), std=torch.FloatTensor(mat_std).cuda())
        self.mtran.cuda()

    def measure_data(self):
        # Loop through dataset using augmentation to measure Mean and Standard Deviation
        matrix_all = []
        for real, matrix in tqdm(self.train_loader):
            matrix_all.append(matrix.numpy())
        matrix_array = np.concatenate(matrix_all, axis=0)
        return matrix_array.mean(0), matrix_array.std(0)

    def load_state(self, filepath):
        # Load previously saved sate from disk, including models, optimizers and history
        state = torch.load(filepath)
        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1

        for i in self.model_dict.keys():
            self.model_dict[i].load_state_dict(state['models'][i])
        for i in self.opt_dict.keys():
            self.opt_dict[i].load_state_dict(state['optimizers'][i])

        self.train_hist_dict = state['train_hist']
        self.train_hist_dict_test = state['train_hist_test']
        self.display_history()

    def save_state(self, filepath):
        # Save current state of all models, optimizers and history to disk
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()

        model_state = {'iter': self.current_iter,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict,
                       'train_hist_test': self.train_hist_dict_test
                       }

        torch.save(model_state, filepath)
        return f'Saving State at Iter:{self.current_iter}'

    def display_history(self):
        # Draw history of losses, called at end of training
        fig = plt.figure()
        plt.ylim(top=1.5)
        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            x_test = range(len(self.train_hist_dict_test[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)
                plt.plot(x_test, self.train_hist_dict_test[key], label=key + '_test')

        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.legend(loc=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{self.params["save_root"]}_loss.jpg')
        plt.show()
        plt.close(fig)

    def lr_lookup(self):
        # Determine proper learning rate multiplier for this iter, cuts in half every "lr_drop_every"
        div = max(0, ((self.current_epoch - self.params["lr_drop_start"]) // self.params["lr_drop_every"]))
        lr_mult = 1 / math.pow(2, div)
        return lr_mult

    def train_gen(self, matrix, real):
        self.set_grad("G", True)
        self.opt_dict["G"].zero_grad()

        # generate fake
        fake = self.model_dict["G"](matrix)

        # get content loss
        ct_losses, l1_losses = self.ct_loss(self.vgg_tran(fake), self.vgg_tran(real))

        self.loss_batch_dict['L1_Loss'] = sum(l1_losses)
        self.loss_batch_dict['C_Loss'] = sum(ct_losses)

        total_loss = self.loss_batch_dict['L1_Loss'] + self.loss_batch_dict['C_Loss']
        total_loss.backward()

        self.opt_dict["G"].step()
        return fake.detach()

    def test_gen(self, matrix, real):
        # generate fake
        fake = self.model_dict["G"](matrix)

        # get content loss
        ct_losses, l1_losses = self.ct_loss(self.vgg_tran(fake), self.vgg_tran(real))

        self.loss_batch_dict_test['L1_Loss'] = sum(l1_losses)
        self.loss_batch_dict_test['C_Loss'] = sum(ct_losses)

    def set_grad(self, model, grad):
        for param in self.model_dict[model].parameters():
            param.requires_grad = grad

    def test_loop(self):
        # Test on validation set
        self.model_dict["G"].eval()
        self.set_grad("G", False)

        for loss in self.losses:
            self.loss_epoch_dict_test[loss] = []

        # test loop #
        for real, matrix in tqdm(self.test_loader):
            matrix = Variable(matrix).cuda()
            matrix = self.mtran(matrix)
            real = Variable(real).cuda()

            # TEST GENERATOR
            self.test_gen(matrix, real)

            # Append all losses in loss dict #
            [self.loss_epoch_dict_test[loss].append(self.loss_batch_dict_test[loss].item()) for loss in self.losses]
        [self.train_hist_dict_test[loss].append(helper.mft(self.loss_epoch_dict_test[loss])) for loss in self.losses]

    def train_loop(self):
        # Train on train set
        self.model_dict["G"].train()
        self.set_grad("G", True)

        for loss in self.losses:
            self.loss_epoch_dict[loss] = []

        # Set learning rate
        lr_mult = self.lr_lookup()
        self.opt_dict["G"].param_groups[0]['weight_decay'] = self.params['weight_decay']
        self.opt_dict["G"].param_groups[0]['lr'] = lr_mult * self.params['lr']

        # print LR and weight decay
        print(f"Sched Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
        [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}",
               f" Weight Decay:{ self.opt_dict[opt].param_groups[0]['weight_decay']}")
         for opt in self.opt_dict.keys()]

        # train loop
        for real, matrix in tqdm(self.train_loader):
            matrix = Variable(matrix).cuda()
            matrix = self.mtran(matrix)
            real = Variable(real).cuda()

            # TRAIN GENERATOR
            fake = self.train_gen(matrix, real)
            # append all losses in loss dict
            [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].item()) for loss in self.losses]
            self.current_iter += 1
        [self.train_hist_dict[loss].append(helper.mft(self.loss_epoch_dict[loss])) for loss in self.losses]

    def train(self):
        # Train following learning rate schedule
        params = self.params
        while self.current_epoch < params["train_epoch"]:
            epoch_start_time = time.time()

            # TRAIN LOOP
            self.train_loop()

            # TEST LOOP
            self.test_loop()

            # generate test images and save to disk
            if self.current_epoch % params["save_img_every"] == 0:
                helper.show_test(params,
                                 self.transform,
                                 self.mtran,
                                 self.train_data,
                                 self.test_data,
                                 self.model_dict['G'],
                                 save=f'output/{params["save_root"]}_val_{self.current_epoch}.jpg')

            # save
            if self.current_epoch % params["save_every"] == 0:
                save_str = self.save_state(f'output/{params["save_root"]}_{self.current_epoch}.json')
                tqdm.write(save_str)

                # make animated gif
                helper.test_repo(self, self.repo_data, f'output/{params["save_root"]}_{self.current_epoch}.gif')
                self.display_history()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print(f'Epoch Training Training Time: {per_epoch_ptime}')
            [print(f'Train {loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]
            [print(f'Val {loss}: {helper.mft(self.loss_epoch_dict_test[loss])}') for loss in self.losses]
            print('\n')
            self.current_epoch += 1

        self.display_history()
        print('Hit End of Learning Schedule!')
