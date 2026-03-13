#! /user/bin/env/bash


dataset_names=(
	COADREAD
	KIRP
	KIRC
	LIHC
)


for dataset_name in "${dataset_names[@]}"; do
		python extract_mif_features_direct.py \
			--input_dir /home/qiyuan/sdc/CONCH/TCGA_${dataset_name}/x40_0_512/patches_128 \
			--slide_dir /media/qiyuan/Getea/TCGA-WSI/TCGA_${dataset_name}/x40 \
			--output_dir /home/qiyuan/sdc/CONCH/TCGA_${dataset_name}/x40_0_512/ROISE \
			--csv_path /home/qiyuan/sdc/DJ/ROISE/datasets/${dataset_name}_processed.csv \
			--aggregate_factor 16
		python extract_mif_features_direct.py \
			--input_dir /home/qiyuan/sdc/CONCH/TCGA_${dataset_name}/x20_0_256/patches_128 \
			--slide_dir /media/qiyuan/Getea/TCGA-WSI/TCGA_${dataset_name}/x20 \
			--output_dir /home/qiyuan/sdc/CONCH/TCGA_${dataset_name}/x20_0_256/ROISE \
			--csv_path /home/qiyuan/sdc/DJ/ROISE/datasets/${dataset_name}_processed.csv \
			--aggregate_factor 4
done