from pathlib import Path
from NaroNet.utils.DatasetParameters import parameters
from NaroNet.Patch_Contrastive_Learning.patch_contrastive_learning import patch_contrastive_learning
from NaroNet.Patch_Contrastive_Learning.preprocess_images import preprocess_images
from NaroNet.architecture_search.architecture_search import architecture_search
from NaroNet.NaroNet import run_NaroNet
from NaroNet.NaroNet_dataset import get_BioInsights

def main(path):
    # Select Experiment parameters
    params = parameters(path, 'Value')
    possible_params = parameters(path, 'Object')
    best_params = parameters(path, 'Index')    

    # Preprocess Images
    preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])



if __name__ == "__main__":
    path = 'dataset/'
    main(path)
 