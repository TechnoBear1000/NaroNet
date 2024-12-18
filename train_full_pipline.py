import os
import pandas as pd
import torch
from NaroNet.utils.DatasetParameters import parameters
from NaroNet.Patch_Contrastive_Learning.preprocess_images import preprocess_images
from NaroNet.Patch_Contrastive_Learning.patch_contrastive_learning import patch_contrastive_learning
from NaroNet.NaroNet import run_NaroNet
from NaroNet.NaroNet_dataset import get_BioInsights

# Set the experiment path
experiment_path = "/data/DATASET_DATA_DIR"

# Advanced parameters for training
batch_size = 32                # Increased batch size for training
number_of_epochs = 50          # Number of training epochs
pooling_steps = 3              # Increased pooling steps
percentage_validation = 20     # 20% of data for validation
initial_learning_rate = 0.001  # Starting learning rate

patch_width = 512              # Adjust for image resolution if needed
patch_height = 512

# Initialize parameters with settings for training
params = parameters(experiment_path, 'Value')  # Ensure 'Value' corresponds to your setup

# Override default parameters with custom settings
params['batch_size'] = batch_size
params['number_of_epochs'] = number_of_epochs
params['pooling_steps'] = pooling_steps
params['percentage_validation'] = percentage_validation
params['initial_learning_rate'] = initial_learning_rate
params['patch_width'] = patch_width
params['patch_height'] = patch_height

# Set analysis_only flag to False for training
params['analysis_only'] = False

# Specify device
params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Specify path to save pre-trained weights
params['pretrained_weights'] = "/data/NaroNet_models/model.pth"

# Preprocess images with appropriate transformations for training
preprocess_images(
    experiment_path,
    ZScoreNormalization=params.get('PCL_ZscoreNormalization', False),
    patch_size=params.get('PCL_patch_size', 512)
)

# Run Patch Contrastive Learning
patch_contrastive_learning(experiment_path, params)

# Run NaroNet for model training and analysis
run_NaroNet(experiment_path, params)

# Run BioInsights for post-analysis
get_BioInsights(experiment_path, params)
