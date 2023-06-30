# %%
from datasets import load_dataset

dataset_name = "dataset_name"  # Replace with the name of the dataset you want to download
save_directory = "data"  # Replace with the desired save directory

# Load the dataset
dataset = load_dataset('poloclub/diffusiondb', 'large_first_100k')

# Save the dataset to the specified directory
dataset.save_to_disk(save_directory)


