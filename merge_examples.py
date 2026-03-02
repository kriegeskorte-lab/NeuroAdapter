import os
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import glob
from tqdm import tqdm

root_folder = 'figures'
folder_names = ['1', '2', '5', '7']

# Create a combined figure for each folder
for folder_name in tqdm(folder_names, desc="Processing folders"):
    folder_path = os.path.join(root_folder, folder_name)
    if os.path.exists(folder_path):
        # Collect image pairs from this folder
        folder_pairs = []
        pred_files = glob.glob(os.path.join(folder_path, '*_decoded.png'))
        for pred_file in pred_files:
            # Extract ID and construct target filename
            base_name = os.path.basename(pred_file)
            img_id = base_name.replace('_decoded.png', '')
            target_file = os.path.join(folder_path, f'{img_id}_original.png')
            
            # Check if target exists
            if os.path.exists(target_file):
                folder_pairs.append((target_file, pred_file))
        
        # Randomly select up to 50 pairs from this folder
        selected_pairs = random.sample(folder_pairs, min(50, len(folder_pairs)))
        
        # Create figure with custom GridSpec layout
        fig = plt.figure(figsize=(20, 20))
        
        # Create GridSpec with 10 rows and 10 columns
        # Use height_ratios to make pairs tight and add spacing between pairs
        # Rows 0,1 = pair 1 (tight), gap, rows 2,3 = pair 2 (tight), gap, etc.
        height_ratios = []
        for i in range(5):  # 5 pairs
            height_ratios.extend([1, 1])  # target and predicted rows (equal height)
            if i < 4:  # Add spacing between pairs (except after last pair)
                height_ratios.append(0.2)  # spacing row
        
        # Total rows: 5 pairs * 2 rows + 4 spacing rows = 14 rows
        gs = GridSpec(len(height_ratios), 10, figure=fig, 
                     height_ratios=height_ratios, 
                     hspace=0, wspace=0)
        
        # Fill the figure
        for set_idx in range(5):  # 5 sets of 10 pairs each
            # Calculate actual row indices accounting for spacing rows
            target_row_idx = set_idx * 3  # 0, 3, 6, 9, 12
            pred_row_idx = set_idx * 3 + 1  # 1, 4, 7, 10, 13
            
            for col in range(10):
                pair_idx = set_idx * 10 + col
                
                if pair_idx < len(selected_pairs):
                    target_path, pred_path = selected_pairs[pair_idx]
                    
                    # Create target subplot
                    ax_target = fig.add_subplot(gs[target_row_idx, col])
                    target_img = mpimg.imread(target_path)
                    ax_target.imshow(target_img)
                    ax_target.axis('off')
                    
                    # Create predicted subplot
                    ax_pred = fig.add_subplot(gs[pred_row_idx, col])
                    pred_img = mpimg.imread(pred_path)
                    ax_pred.imshow(pred_img)
                    ax_pred.axis('off')
        
        # Save the combined figure for this folder
        output_path = os.path.join(root_folder, f'combined_examples_folder_{folder_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Combined figure for folder {folder_name} saved to: {output_path}")
        print(f"Pairs found in folder {folder_name}: {len(folder_pairs)}")
        print(f"Pairs used in folder {folder_name}: {len(selected_pairs)}")
        print()

print("All combined figures created successfully!")



