import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import DataParallel
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
import openslide
from torch.utils.data import DataLoader

from Adco.model.AdCo import AdCo, Adversary_Negatives

from Adco.training.train_utils import adjust_learning_rate,save_checkpoint
from Adco.training.train import train, init_memory
from data_processing.transform import TwoCropsTransform, GaussianBlur


def main_worker(gpu, args):
    args.gpu = gpu
    print(vars(args))

    device = torch.device('cuda')
    args.device = device

    print("=> creating model '{}'".format(args.arch))
    multi_crop = args.multi_crop
    Memory_Bank = Adversary_Negatives(args.cluster, args.moco_dim, multi_crop).to(device)

    model = AdCo(args, args.moco_dim, args.moco_m, args.moco_t, args.mlp).to(device)

    model = DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # cudnn.benchmark = True
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError("default csv_path is not implemented!")

    print(csv_path)
    bags_dataset = Dataset_All_Bags(csv_path)
    total = len(bags_dataset)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    # generate train_loaders
    train_loaders = []
    total_length = 0
    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx]
        bag_name = slide_id + '.h5'
        # coord file path
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', args.data_type, bag_name)

        slide_filename = slide_id + ".svs"
        slide_file_path = os.path.join(args.data_slide_dir, slide_filename)
        wsi = openslide.open_slide(slide_file_path)

        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, pretrained=True,
                                     transform=TwoCropsTransform(transforms.Compose(augmentation)))

        kwargs = {'num_workers': 25, 'pin_memory': True} if device.type == "cuda" else {}
        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs)
        total_length += len(dataset)
        train_loaders.append(train_loader)

    print(f"a total of {total_length} patches")
    model.eval()
    #init memory bank
    if args.ad_init and not os.path.isfile(args.resume):
        init_memory(train_loaders, model, Memory_Bank, criterion,
              optimizer, 0, total_length, args)
        print("Init memory bank finished!!")
    best_Acc=0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1 = train(train_loaders, model, Memory_Bank, criterion,
                     optimizer, epoch, total_length, args)
        best_Acc=max(best_Acc, acc1)

    save_dict = {
        'arch': args.arch,
        'best_acc': best_Acc,
        'state_dict': model.state_dict(),
    }
    tmp_save_path = os.path.join(args.save_path, f'adco_{args.data_type}_not_sym.pth.tar')

    save_checkpoint(save_dict, filename=tmp_save_path)
