import pandas as pd
import ast
import numpy as np
from tqdm import tqdm

file_path = "./logs/syntheticstatic_batchsize-16384/patch_logger.csv"
df = pd.read_csv(file_path)
patches_in_batch = 1024 #df.iloc[0]["Patches in Batch"]
patch_size = df.iloc[0]["Patch Size"]

# Group patches that belong together in a batch
batch_groups = df.groupby(df.index // patches_in_batch)

grouped_dfs = {}
for group_name, group_df in batch_groups:
    grouped_dfs[group_name] = group_df.reset_index(drop=True)

uniqueness_percentages = []

for i, batch_group in tqdm(grouped_dfs.items()):
    pixels_in_batch = patches_in_batch * patch_size * patch_size
    unique_pixels_in_batch = 0

    image_groups = batch_group.groupby('Image Index')
    for j, image_group in image_groups:
        count, columns = image_group.shape
        if count > 1:
            all_pixels = []
            for _, row in image_group.iterrows():
                patch = ast.literal_eval(row['Pixel Indices'])
                all_pixels.extend(patch)

            unique_pixels = np.unique(all_pixels)
            unique_pixels_in_batch += len(unique_pixels)
        else:
            unique_pixels_in_batch += patch_size * patch_size
    
    uniqueness_percentages.append(unique_pixels_in_batch / pixels_in_batch)

# Per batch percentage of unique pixels
num_batches = len(grouped_dfs.items())
uniqueness_per_batch = sum(uniqueness_percentages) / num_batches
print(f"Per batch percentage of unique pixels is: {uniqueness_per_batch * 100}%")