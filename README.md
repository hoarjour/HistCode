# HistCode
Source code and data for "Contrastive learning-based computational histopathology predict differential expression of cancer driver genes"

# Now Updating

## Seg and Tile
You can download your own wsi dataset to the directory slides, then run data_processing/create_patches_fp.py to seg and tile wsis, adjust the parameters according to your needs

## Train Contrast Learning Model
run train/train_adco.py to train contrast learning model on tiles, you should write Adco/ops/argparser.py to configure the data source and the save address and ADCO related parameters firstly

