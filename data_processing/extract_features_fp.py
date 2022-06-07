import torch
import torch.nn as nn
from collections import OrderedDict
import os
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from utils.utils import collate_features
from utils.file_utils import save_hdf5
import openslide
import time
from networks.resnet_custom import resnet50_baseline

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size,
                                 train=False)
    kwargs = {'num_workers': 32, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='../tile_results')
parser.add_argument('--data_slide_dir', type=str, default="../slides/TCGA-LUNG")
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default='../dataset_csv/sample_data.csv')
parser.add_argument('--feat_dir', type=str, default='../features')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)

parser.add_argument('--model_type', type=str, default="ADCO")
parser.add_argument('--data_type', type=str, default="tcga_lung")
parser.add_argument('--model_path', type=str, default="../MODELS/adco_tcga_lung_not_sym.pth.tar")
args = parser.parse_args()


def load_model(weight_path, data_parallel=True):
    check_point = torch.load(weight_path)
    state_dict = check_point["state_dict"]
    model = resnet50_baseline()
    if data_parallel:
        prefix = "module.encoder_q."
    else:
        prefix = "encoder_q."
    new_sd = OrderedDict()
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = state_dict[k]
    missing_key = model.load_state_dict(new_sd, strict=False)
    assert set(missing_key.unexpected_keys) == {"fc.0.weight", "fc.0.bias", "fc.2.weight", "fc.2.bias"}
    return model


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    type_path = args.model_type

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files', type_path), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files', type_path), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files', type_path))

    model = load_model(args.model_path)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx][:-3]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', args.data_type, bag_name)
        if not os.path.isfile(h5_file_path):
            print(f"h5 file {h5_file_path} not exist")
            continue

        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        output_path = os.path.join(args.feat_dir, 'h5_files', type_path, bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                            model=model, batch_size=args.batch_size, verbose=1, print_every=5,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
