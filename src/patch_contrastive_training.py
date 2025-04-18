from NaroNet.utils.DatasetParameters import parameters
from NaroNet.Patch_Contrastive_Learning.patch_contrastive_learning import patch_contrastive_learning

def main(path):
    # Select Experiment parameters
    params = parameters(path, 'Value')
    possible_params = parameters(path, 'Object')
    best_params = parameters(path, 'Index')    

    # Preprocess Images
    #preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

    # Patch Contrastive Learning
    patch_contrastive_learning(path, params)
    
    # Architecture Search
    # params = architecture_search(path,params,possible_params)

    #run_NaroNet(path,params)
    
    # BioInsights
    #get_BioInsights(path,params)

if __name__ == "__main__":
    path = 'dataset/'            
    main(path)