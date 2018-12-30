
## Getting Started
- Check your python version, this is built on `python 3.6`
- Install `pytorch 0.4.0` and dependencies from https://pytorch.org/
- Install packages `tqdm`, `cv2`, `matplotlib`, `torchvision`
- Install `ImageMagick` and `rar`

- Clone this repo:
```bash
git clone https://github.com/NeuralVFX/reverse-matchmove-gan.git
cd reverse-matchmove-gan
```
- Download the dataset (e.g. [Chiang Mai](http://neuralvfx.com/datasets/reverse_matchmove/chiang_mai.rar)):
```bash
bash data/get_test_dataset.sh
```

## Train The Model
```bash
python train.py --dataset chiang_mai --train_epoch 200  --save_root chiang_mai
```

## Continue Training Existing Saved State
```bash
python train.py --dataset chian_mai --train_epoch 200  --save_root chiang_mai --load_state output/chiang_mai_3.json
```

## Command Line Arguments
```
--dataset, default='chiang_mai', type=str                      # Dataset folder name
--batch_size, default=5, type=int                              # Training batch size
--workers, default=8, type=int                                 # How many threads to help with dataloading
--res, default=512, type=int                                   # Image resolution, for dataloading, and generator
--vgg_layers_c, default=[2,7,12], type=int                     # Layers of VGG to use for content
--l1_weight, default=3., type=float                            # Multiplier for L1 loss
--content_weight, default=2.5, type=float                      # Multiplier for perceptual loss
--vgg_weight_div, default=1, type=float                        # Multiplier for each next layer in VGG perceptual loss
--train_epoch, default=200, type=int                           # Number of epochs to train for
--beta1, default=.5, type=float                                # Beta1 value used by optimizer
--beta2, default=.999, type=float                              # Beta2 value used by optimizer
--drop, default=.01, type=float                                # Multiplier dropout on later layers of generator
--center_drop, default=.01, type=float                         # Multiplier dropout on first two layers of generator
--lr, default=2e-4, type=float                                 # Learning rate
--lr_drop_start, default=0, type=int                           # Epoch on which the learning rate will begin to drop
--lr_drop_every, default=5, type=int                           # How many epochs between every learning rate drop, learning rate will cut in half each time
--ids_test, default=[1,100], type=int                          # Ids from test set for preview images
--ids_train, default=[0,2], type=int                           # Ids from training set for preview images
--save_every, default=5, type=int                              # How many epochs between each model save
--save_img_every, default=1, type=int                          # How many epochs between saving image
--save_root, default='chiang_mai', type=str                    # Prefix for files created by the model under the /output directory
--load_state, type=str                                         # Optional: filename of state to load and resume training from
```

## Data Folder Structure

- Dataset:

`data/<data set>/`

- Train Dir:

`data/imagenet/`

## Output Folder Structure

- `weights`, `test images`, `test animated gif` and `loss graph`, are all output to this directory: `output/<save_root>_*.*`
