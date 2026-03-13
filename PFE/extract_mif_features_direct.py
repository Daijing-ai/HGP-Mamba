import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import torchvision.models as models
from torch.utils.data import Dataset
import pandas as pd
import openslide
import h5py
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

# Configuration constants
BATCH_SIZE = 256
NUM_WORKERS = 8

def get_model(num_outputs: int) -> nn.Module:
    """Creates and returns the model architecture."""
    model = models.convnext_small(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model

class Dataset_All_Bags(Dataset):
		
	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = 0
			self.patch_size = 512
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		img = np.array(img)
		if isinstance(self.roi_transforms, dict):
			he_patch_pt = self.roi_transforms['all_channels'](img)
			patch = self.roi_transforms['image_only'](he_patch_pt)
		else:
			patch = self.roi_transforms(img)
		return patch

def save_hdf5(output_path, feature_vector, mode='a'):
    """
    Save a single patch feature vector (shape: [50]) to an HDF5 file.
    The dataset 'features' will have shape [n, 50], where n is the number of patches.
    """
    feature_vector = np.asarray(feature_vector, dtype=np.float32)
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)  # shape: [1, 50]
    with h5py.File(output_path, mode) as file:
        if 'features' not in file:
            dset = file.create_dataset(
                'features',
                shape=(0, feature_vector.shape[1]),
                maxshape=(None, feature_vector.shape[1]),
                chunks=(32, feature_vector.shape[1]),
                dtype=np.float32
            )
        else:
            dset = file['features']
        # Resize and append
        old_n = dset.shape[0]
        new_n = old_n + feature_vector.shape[0]
        dset.resize(new_n, axis=0)
        dset[old_n:new_n, :] = feature_vector
    return output_path


def main():
		parser = argparse.ArgumentParser(description='Run inference on H&E images')
		parser.add_argument('--input_dir', type=str, default='/home/qiyuan/sdc/CONCH/TCGA_COADREAD/x40_0_512/patches_128', help='Directory containing h5 files')
		parser.add_argument('--slide_dir', type=str, default='/media/qiyuan/Getea/TCGA-WSI/TCGA_COADREAD/x40', help='Directory containing slide files')
		parser.add_argument('--output_dir', type=str, default='/home/qiyuan/sdc/CONCH/TCGA_COADREAD/x40_0_512/ROISE', help='Directory to save output features')
		parser.add_argument('--model_path', type=str, default='best_model_single.pth', help='Path to trained model weights')
		parser.add_argument('--csv_path', type=str, default='/home/qiyuan/sdc/DJ/ROISE/datasets/COADREAD_processed.csv', help='Path to CSV file containing slide information')
		parser.add_argument('--aggregate_factor', type=int, default=16, help='Factor to aggregate features')
		args = parser.parse_args()


		h5_dir = os.path.join(args.output_dir, 'h5_files')
		os.makedirs(h5_dir, exist_ok=True)
		
		# Set up device
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Initialize dataset
		print('initializing dataset...')
		csv_path = args.csv_path
		if csv_path is None:
			raise ValueError("CSV path must be provided")
		bags_dataset = Dataset_All_Bags(csv_path)
		total = len(bags_dataset)
		print(f'Total slides to process: {total}')

		# Load model
		num_channels = 50
		model = get_model(num_outputs=num_channels)
		# if torch.cuda.device_count() > 1:
		#			print("Using", torch.cuda.device_count(), "GPUs")
		# pdb.set_trace()
		checkpoint = torch.load(args.model_path)['model_state_dict']
		new_checkpoint = OrderedDict()
		for k, v in checkpoint.items():
			new_k = k.replace('module.', '', 1)
			new_checkpoint[new_k] = v

		model.load_state_dict(new_checkpoint)
		model = model.to(device)

		# Define transformations
		img_transforms = {
					'all_channels': transforms.Compose([
							transforms.ToTensor(),
							transforms.Resize(224, antialias=True),
					]),
					'image_only': transforms.Compose([
							transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
					])
			}

		for bag_idx in tqdm(range(total)):
			slide_id = bags_dataset[bag_idx]
			bag_name = slide_id + '.h5'
			h5_file_path = os.path.join(args.input_dir, bag_name)
			if not os.path.exists(h5_file_path):
				print(f"Warning: {h5_file_path} does not exist. Skipping.")
				continue
			slide_file_path = os.path.join(args.slide_dir, bag_name.replace('.h5', '.svs'))
			print('\nprogress: {}/{}...'.format(bag_idx, total))
			print(slide_id)
			
			if bag_name in os.listdir(h5_dir):
				print(f"Skipping {bag_name} - already processed")
				continue
			
			output_path = os.path.join(h5_dir, bag_name)
			
			wsi = openslide.open_slide(slide_file_path)
			dataset = Whole_Slide_Bag_FP(file_path=h5_file_path,
																	wsi=wsi,
																	img_transforms=img_transforms)
			
			dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
			model.eval()
			with torch.no_grad():
				for patches in tqdm(dataloader):
					patches = patches.to(device)
					predictions = model(patches).cpu().numpy()
					# Save features to h5 file (append each patch's feature vector)
					save_hdf5(output_path, predictions, mode='a')
			
			# read features from output h5
			with h5py.File(output_path, "r") as file:
				features = file['features'][:]
			# coords should come from the source h5 (h5_dir), not the output file
			with h5py.File(h5_file_path, "r") as src:
				coords = src['coords'][:] if 'coords' in src else None

			# Aggregate features if needed (assume divisibility)
			if args.aggregate_factor > 1:
				factor = args.aggregate_factor
				n_patches = features.shape[0]
				assert n_patches % factor == 0, f"n_patches ({n_patches}) must be divisible by factor ({factor})"
				n_aggregated = n_patches // factor
				# average features per group
				features_512 = features.reshape(n_aggregated, factor, -1).mean(axis=1)
				if coords is not None:
					coords_reshaped = coords.reshape(n_aggregated, factor, -1)
					coords_512 = coords_reshaped[:, 0, :]

				# write aggregated datasets to output h5
				with h5py.File(output_path, 'a') as file:
					file.create_dataset('features_512', data=features_512, dtype=np.float32)
					file.create_dataset('coords_512', data=coords_512)

			# save per-bag .pt file
			pt_dir = os.path.join(args.output_dir, 'pt_files')
			os.makedirs(pt_dir, exist_ok=True)
			features_tensor = torch.from_numpy(features_512)
			bag_base, _ = os.path.splitext(bag_name)
			torch.save(features_tensor, os.path.join(pt_dir, bag_base + '.pt'))

if __name__ == '__main__':
	main()