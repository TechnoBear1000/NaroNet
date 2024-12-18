import argparse
import numpy as np
from hyperopt import hp

#### changes made ####

# args['PCL_embedding_dimensions'] = 256
# args['PCL_batch_size'] = 160
# args['PCL_depth_CNN'] = 50
# args['PCL_epochs'] = 200
# args['dataAugmentationPerc'] = 0.001
# args['folds'] = 5
# args['batch_size'] = 4 
# args['clusters1'] = 6 (phenotypes)
# args['clusters2'] = 7 (neighborhoods)
# args['clusters3'] = 4 (areas)
# args['lr'] = 0.001
# args['weight_decay'] = 0.01

# args['experiment_Label'] = ['Placeholder']


def parameters(path, debug):
    args = {}
    args['path'] = path

    # Patch contrastive learning parameters
    args['PCL_embedding_dimensions'] = 256
    args['PCL_batch_size'] = 160
    args['PCL_epochs'] = 200
    args['PCL_patch_size'] = 15
    args['PCL_alpha_L'] = 1.2  # The value of alpha_L in the manuscript
    args['PCL_ZscoreNormalization'] = True
    args['PCL_width_CNN'] = 2  # [1, 2, 4]
    args['PCL_depth_CNN'] = 50  # [1, 2, 4]

    # Label you want to infer with respect to the images
    args['experiment_Label'] = ['Placeholder']


    # Optimization Parameters
    args['num_samples_architecture_search'] = 2
    args['epochs'] = 10  # if debug else hp.quniform('epochs', 5, 25, 1)
    args['epoch'] = 0
    args['lr_decay_factor'] = 0.5  # if debug else hp.uniform('lr_decay_factor', 0, 0.75)
    args['lr_decay_step_size'] = 12  # if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)
    args['weight_decay'] = 0.01
    args['batch_size'] = 4
    

    args['lr'] = 0.001
    
    args['useOptimizer'] = 'ADAM'  # 0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound'])

    # General
    args['context_size'] = 15  # 0 if debug else hp.choice("context_size", [15])
    args['num_classes'] = 3
    args['MultiClass_Classification'] = 1
    args['showHowNetworkIsTraining'] = False  # Creates a GIF of the learning clusters!
    args['visualizeClusters'] = True
    args['learnSupvsdClust'] = True
    args['recalculate'] = False
    args['folds'] = 5
    args['device'] = 'cuda:3'
    args['normalizeFeats'] = (
        1 if debug == 'Index' 
        else hp.choice("normalizeFeats", [True, False]) 
        if debug == 'Object' 
        else False
    )
    args['normalizeCells'] = (
        1 if debug == 'Index' 
        else hp.choice("normalizeCells", [True, False]) 
        if debug == 'Object' 
        else False
    )
    args['Batch_Normalization'] = (
        0 if debug == 'Index' 
        else hp.choice("Batch_Normalization", [True, False]) 
        if debug == 'Object' 
        else True
    )
    args['normalizePercentile'] = False  # 1 if debug else hp.choice("normalizePercentile", [True, False])
    args['dataAugmentationPerc'] = (
        1 if debug == 'Index' 
        else hp.choice("dataAugmentationPerc", [0, 0.0001, 0.001, 0.01, 0.1]) 
        if debug == 'Object' 
        else 0.0001
    )

    # Neural Network
    args['hiddens'] = (
        1 if debug == 'Index' 
        else hp.choice('hiddens', [32, 44, 64, 86, 128]) 
        if debug == 'Object' 
        else 44
    )
    
    args['clusters1'] = 6
    args['clusters2'] = 7
    args['clusters3'] = 4

    args['LSTM'] = False  # 0 if debug else hp.choice("LSTM", [True, False])
    args['GLORE'] = (
        1 if debug == 'Index' 
        else hp.choice('GLORE', [True, False]) 
        if debug == 'Object' 
        else False
    )
    args['Phenotypes'] = True
    args['DeepSimple'] = False
    args['isAttentionLayer'] = False  # 1 if debug else hp.choice("isAttentionLayer", [True, False])
    args['ClusteringOrAttention'] = (
        0 if debug == 'Index' 
        else hp.choice("ClusteringOrAttention", [True, False]) 
        if debug == 'Object' 
        else True
    )
    args['1cell1cluster'] = (
        1 if debug == 'Index' 
        else hp.choice("1cell1cluster", [True, False]) 
        if debug == 'Object' 
        else False
    )
    args['dropoutRate'] = (
        3 if debug == 'Index' 
        else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2, 0.25]) 
        if debug == 'Object' 
        else 0.2
    )
    args['AttntnSparsenss'] = False  # 1 if debug else hp.choice("AttntnSparsenss", [True, False])
    args['attntnThreshold'] = (
        0 if debug == 'Index' 
        else hp.choice('attntnThreshold', [0, .2, .4, .6, .8]) 
        if debug == 'Object' 
        else 0
    )
    args['GraphConvolution'] = (
        0 if debug == 'Index' 
        else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) 
        if debug == 'Object' 
        else 'ResNet'
    )
    args['n-hops'] = (
        2 if debug == 'Index' 
        else hp.choice('n-hops', [1, 2, 3]) 
        if debug == 'Object' 
        else 3
    )
    args['modeltype'] = 'SAGE'  # 0 if debug else hp.choice('modeltype', ['SAGE', 'SGC'])
    args['ObjectiveCluster'] = True  # 1 if debug else hp.choice('ObjectiveCluster', [True, False])
    args['ReadoutFunction'] = False  # 0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets'])
    args['NearestNeighborClassification'] = False  # 1 if debug else hp.choice('NearestNeighborClassification', [True, False])
    args['NearestNeighborClassification_Lambda0'] = 1  # 0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01, 0.001, 0.0001])
    args['NearestNeighborClassification_Lambda1'] = 1
    args['NearestNeighborClassification_Lambda2'] = 1
    args['KinNearestNeighbors'] = 5

    # Losses
    args['pearsonCoeffSUP'] = False
    args['pearsonCoeffUNSUP'] = False
    args['orthoColor'] = (
        0 if debug == 'Index' 
        else hp.choice("orthoColor", [True, False]) 
        if debug == 'Object' 
        else True
    )
    args['orthoColor_Lambda0'] = (
        0 if debug == 'Index' 
        else hp.choice("orthoColor_Lambda0", [0.1, 0.01, 0.001, 0.0001, 0.00001]) 
        if debug == 'Object' 
        else 0.1
    )
    args['orthoColor_Lambda1'] = (
        4 if debug == 'Index' 
        else hp.choice("orthoColor_Lambda1", [0.1, 0.01, 0.001, 0.0001, 0.00001]) 
        if debug == 'Object' 
        else 0.00001
    )
    args['ortho'] = (
        1 if debug == 'Index' 
        else hp.choice("ortho", [True, False]) 
        if debug == 'Object' 
        else False
    )
    args['ortho_Lambda0'] = (
        0 if debug == 'Index' 
        else hp.choice("ortho_Lambda0", [0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 0.1
    )
    args['ortho_Lambda1'] = (
        4 if debug == 'Index' 
        else hp.choice("ortho_Lambda1", [0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 0
    )
    args['ortho_Lambda2'] = (
        4 if debug == 'Index' 
        else hp.choice("ortho_Lambda2", [0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 0
    )
    args['min_Cell_entropy'] = (
        0 if debug == 'Index' 
        else hp.choice("min_Cell_entropy", [True, False]) 
        if debug == 'Object' 
        else True
    )
    args['min_Cell_entropy_Lambda0'] = (
        0 if debug == 'Index' 
        else hp.choice("min_Cell_entropy_Lambda0", [1, 0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 1
    )
    args['min_Cell_entropy_Lambda1'] = (
        4 if debug == 'Index' 
        else hp.choice("min_Cell_entropy_Lambda1", [1, 0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 0.0001
    )
    args['min_Cell_entropy_Lambda2'] = (
        2 if debug == 'Index' 
        else hp.choice("min_Cell_entropy_Lambda2", [1, 0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 0.01
    )
    args['MinCut'] = (
        0 if debug == 'Index' 
        else hp.choice("MinCut", [True, False]) 
        if debug == 'Object' 
        else True
    )
    args['MinCut_Lambda0'] = (
        5 if debug == 'Index' 
        else hp.choice("MinCut_Lambda0", [1, 0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 0
    )
    args['MinCut_Lambda1'] = (
        1 if debug == 'Index' 
        else hp.choice("MinCut_Lambda1", [1, 0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 0.1
    )
    args['MinCut_Lambda2'] = (
        1 if debug == 'Index' 
        else hp.choice("MinCut_Lambda2", [1, 0.1, 0.01, 0.001, 0.0001, 0]) 
        if debug == 'Object' 
        else 0.1
    )
    args['F-test'] = False
    args['Max_Pat_Entropy'] = (
        1 if debug == 'Index' 
        else hp.choice('Max_Pat_Entropy', [True, False]) 
        if debug == 'Object' 
        else False
    )
    args['Max_Pat_Entropy_Lambda0'] = (
        4 if debug == 'Index' 
        else hp.choice("Max_Pat_Entropy_Lambda0", [1, 0.1, 0.01, 0.001, 0.0001]) 
        if debug == 'Object' 
        else 0.0001
    )
    args['Max_Pat_Entropy_Lambda1'] = (
        1 if debug == 'Index' 
        else hp.choice("Max_Pat_Entropy_Lambda1", [1, 0.1, 0.01, 0.001, 0.0001]) 
        if debug == 'Object' 
        else 0.1
    )
    args['Max_Pat_Entropy_Lambda2'] = (
        1 if debug == 'Index' 
        else hp.choice("Max_Pat_Entropy_Lambda2", [1, 0.1, 0.01, 0.001, 0.0001]) 
        if debug == 'Object' 
        else 0.1
    )
    args['UnsupContrast'] = False
    args['UnsupContrast_Lambda0'] = 0
    args['UnsupContrast_Lambda1'] = 0
    args['UnsupContrast_Lambda2'] = 0
    args['Lasso_Feat_Selection'] = (
        1 if debug == 'Index' 
        else hp.choice("Lasso_Feat_Selection", [True, False]) 
        if debug == 'Object' 
        else False
    )
    args['Lasso_Feat_Selection_Lambda0'] = (
        1 if debug == 'Index' 
        else hp.choice("Lasso_Feat_Selection_Lambda0", [1, 0.1, 0.01, 0.001, 0]) 
        if debug == 'Object' 
        else 0.1
    )
    args['SupervisedLearning_Lambda0'] = (
        0 if debug == 'Index' 
        else hp.choice("SupervisedLearning_Lambda0", [1, 0.1, 0.01, 0.001, 0.0001]) 
        if debug == 'Object' 
        else 1
    )
    args['SupervisedLearning_Lambda1'] = (
        0 if debug == 'Index' 
        else hp.choice("SupervisedLearning_Lambda1", [1, 0.1, 0.01, 0.001, 0.0001]) 
        if debug == 'Object' 
        else 1
    )
    args['SupervisedLearning_Lambda2'] = (
        0 if debug == 'Index' 
        else hp.choice("SupervisedLearning_Lambda2", [1, 0.1, 0.01, 0.001, 0.0001]) 
        if debug == 'Object' 
        else 1
    )
    args['SupervisedLearning_Lambda3'] = (
        0 if debug == 'Index' 
        else hp.choice("SupervisedLearning_Lambda3", [1, 0.1, 0.01, 0.001, 0.0001]) 
        if debug == 'Object' 
        else 1
    )
    args['SupervisedLearning'] = True  # if debug else hp.choice("SupervisedLearning", [True, False])

    return args
