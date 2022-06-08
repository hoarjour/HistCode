import argparse
import os

# internal imports
from utils.utils import *
import torch
from datasets.dataset_gene import Gene_Reg_Dataset, Multi_Gene_Reg_Dataset
from networks.model_gene_reg import CLAM_SB_Reg, EarlyStop, CLAM_SB_Reg_NN_Pool

parser = argparse.ArgumentParser(description='Configurations for gene Training')
parser.add_argument('--data_root_dir', type=str, default="../FEATURES",
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--extract_model', type=str, default="ADCO")
parser.add_argument('--results_dir', type=str, default='../MODELS/gene', help='results directory (default: ../MODELS/gene)')
parser.add_argument('--csv_dir', type=str, default='../dataset_csv/gene')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')

args = parser.parse_args()
encoding_size = 1024
args.n_classes = 1


if __name__ == '__main__':
    device = torch.device("cuda")
    for i in range(args.k):
        print(f"fold {i + 1}")
        print('\nInit dataset...', end=' ')
        h5_dir = os.path.join(args.data_root_dir, "h5_files", args.extract_model)

        train_dataset = Multi_Gene_Reg_Dataset(h5_dir=h5_dir,
                                               csv_path=f"{args.csv_dir}/train_dataset_{i}.csv",
                                               norm=True)
        test_dataset = Multi_Gene_Reg_Dataset(h5_dir=h5_dir,
                                              csv_path=f"{args.csv_dir}/test_dataset_{i}.csv",
                                              test=True,
                                              norm=True)

        print('Done!')

        all_gene_names = train_dataset.get_all_gene_names()

        for gene_name in all_gene_names:
            # change to gene_name
            train_dataset.switch(gene_name)
            test_dataset.switch(gene_name)
            # train loader
            kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}
            train_loader = DataLoader(dataset=train_dataset, batch_size=1, **kwargs)

            # test loader
            test_loader = DataLoader(dataset=test_dataset, batch_size=1, **kwargs)

            loss_fn = nn.MSELoss().to(device)

            print('\nInit Model...', end=' ')
            model = CLAM_SB_Reg(n_classes=args.n_classes)
            print("Done!")

            model.relocate(device)
            optimizer = get_optim(model, args)

            # model save path
            save_dir = os.path.join(args.results_dir, gene_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, f"checkpoint_{i + 1}_ADCO_atten_tcga.pt")

            early_stop = EarlyStop(persistence=10, path=save_path)

            for epoch in range(args.max_epochs):
                # train
                model.train()
                train_loss = 0.

                for batch_idx, (data, label) in enumerate(train_loader):
                    data, label = torch.squeeze(data), torch.squeeze(label, 0)
                    data_, label_ = data.to(device), label.to(device)

                    logits = model(data_)

                    loss = loss_fn(logits[0], label_)
                    loss_value = loss.item()
                    loss.backward()

                    train_loss += loss_value

                    optimizer.step()
                    optimizer.zero_grad()
                train_loss /= len(train_dataset)
                print('Epoch: {}, train_loss: {:.4f}'.format(epoch, train_loss))

                model.eval()
                val_loss = 0.
                for batch_idx, (data, label) in enumerate(test_loader):
                    data, label = torch.squeeze(data), torch.squeeze(label, 0)
                    data_, label_ = data.to(device), label.to(device)
                    with torch.no_grad():
                        logits = model(data_)

                        loss = loss_fn(logits[0], label_)
                        loss_value = loss.item()
                    val_loss += loss_value
                val_loss /= len(test_dataset)
                print('Epoch: {}, test_loss: {:.4f}'.format(epoch, val_loss))
                early_stop(model, val_loss)

                if early_stop.early_stop:
                    # if overfit
                    print(f"### fold {i + 1} gene {gene_name} overfit，in epoch {epoch} stop train ###")
                    break

                # if not overfit
                if epoch == args.max_epochs - 1:
                    print(f"### fold {i + 1} gene {gene_name} not overfit ###")
                    torch.save(model.state_dict(), save_path)

    print("train successfully")
