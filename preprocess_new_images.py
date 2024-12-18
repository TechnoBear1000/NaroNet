import os
from NaroNet.utils.DatasetParameters import parameters
from NaroNet.Patch_Contrastive_Learning.preprocess_images import preprocess_images

def preprocess_new_images(path):
    params = parameters(path, 'Value')
    image_representation_dir = os.path.join(path, 'Patch_Contrastive_Learning', 'Image_Patch_Representation')

    # Ensure the representation directory exists
    if not os.path.exists(image_representation_dir):
        os.makedirs(image_representation_dir)

    # Add 'image_representation_dir' to params so 'preprocess_images' can access it
    params['image_representation_dir'] = image_representation_dir

    # Call preprocess_images with 'path' and 'params'
    preprocess_images(path, params)

if __name__ == "__main__":
    path = '/mnt/g/Adjuvant_tcells/Adjuvant_tcell_project_NaroNet_training/DATASET_DATA_DIR/'            
    preprocess_new_images(path)
