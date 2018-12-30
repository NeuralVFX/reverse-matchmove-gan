# Reverse-Matchmove-GAN
The goal of this project is to be able to reverse engineer new camera views from existing footage. This is the same network from my blog post: http://neuralvfx.com/matchmove/reverse-matchmove-gan/

My dataset was created by photographing many random angles of a statue in Chiang Mai, then using a photo-modeling tool to extract camera positions and create a CSV file of Matrix-Image pairs. 

The Chiang Mai dataset can be downloaded here: http://neuralvfx.com/datasets/reverse_matchmove/chiang_mai.rar

# Generated Video Example
![](examples/anim_example.gif)

# Code Usage
Usage instructions found here: [user manual page](USAGE.md).

# Example Data Set
![](examples/chiang_mai_matrix_data_b.png)

# Example Augmentation
![](examples/augmentation_a.png)

# Example Results
#### (1: Camera Matrix — 2: Generated Image — 3: Ground Truth Image)
![](examples/chiang_mai_example.png)
