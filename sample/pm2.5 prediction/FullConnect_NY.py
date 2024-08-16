import argparse
import os
import numpy as np
from dataloader import CityDataset
import torch
import torch.nn as nn
import torchvision.models as models
import pickle
from datalayer import *
from metrics import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import *
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import random
import numpy as np
import torch

# random_seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

name = ""


class FC_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC_Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.model(x)
        return x

def train_FC(model, data, epoch, batch_size, lr, model_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = Data.TensorDataset() #TODO
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    for i in range(epoch):
        epoch_loss = 0.0
        epoch_accuracy = []
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_accuracy.append(accuracy(output, y)) #TODO


def main(args, target, features):
    writer = SummaryWriter(
        log_dir="log/FC/{}/{}/{}".format(
            args.date, args.feature, args.epoch
        )
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(len(features), 64),
        nn.ReLU(),
        nn.Linear(64, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    modif_model = model
    if args.load:
        modif_model.load_state_dict(torch.load(args.modelurl))
    modif_model.to(device)
    optimizer = torch.optim.Adam(modif_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    input_feature = combine_feature(*features)  # C * H * W
    data_list = split_data(input_feature, 1, 1)
    intersection_dens_data = combine_feature(target)
    intersection_dens_dataset = split_data(intersection_dens_data, 1, 1)
    dataset = CityDataset(data_list, intersection_dens_dataset)
    dataset_len = len(dataset)
    train_len = int(dataset_len * 0.9)
    test_len = dataset_len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    for epoch in range(args.epoch):
        loss = train(train_loader, modif_model, optimizer, criterion, len(features))
        # print("epoch: %d, loss: %f" % (epoch, loss))
        writer.add_scalar("loss", loss, epoch)
        # scheduler.step()

    mae_score, rmse_score, r_2_score = test(test_loader, modif_model, len(features))
    print(
        # "mae_score: %f, rmse_score: %f, r_2_score: %f"
        "mae_score, rmse_score, r_2_score: %.3f & %.3f & %.3f"
        % (mae_score, rmse_score, r_2_score)
    )
    if args.save:
        path = "model/resnet/{}/{}/".format(
            args.date, args.feature
        )
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            modif_model.state_dict(),
            "model/resnet/" + args.date + "/" + name,
        )
        print("model saved")
    # print(modif_model)


def load_data():
    pm = pickle.load(
        open("data/NewYorkState_pm2.5_imputedS.pickle", "rb")
    )
    pm.print_info()
    nightlight = pickle.load(
        open("data/NewYorkState_light_imputedS.pickle", "rb")
    )
    nightlight.print_info()
    population = pickle.load(
        open("data/NewYorkState_pop_imputedS.pickle", "rb")
    )
    population.print_info()
    intersection_dens = pickle.load(
        open("data/NewYorkState_inter.pickle", "rb")
    )
    intersection_dens.print_info()

    # scaler = StandardScaler()
    # pm.data = scaler.fit_transform(pm.data)
    # nightlight.data = scaler.fit_transform(nightlight.data)
    # population.data = scaler.fit_transform(population.data)
    # intersection_dens.data = scaler.fit_transform(intersection_dens.data)

    return pm, nightlight, population, intersection_dens


def train(loader, model, optimizer, criterion, len_features):
    model.train()
    epoch_loss = 0
    for i, sample in enumerate(loader):
        input_feature = sample["x"].float().reshape(-1,len_features)  # .cuda()
        label = sample["y"].float().reshape(-1,1)
        output = model(input_feature)
        optimizer.zero_grad()

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def test(test_loader, model, len_features):
    model.eval()
    total_output = []
    total_label = []
    loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            input_feature = sample["x"].float().reshape(-1,len_features)  # .cuda()
            label = sample["y"].float().flatten().reshape(-1,1)
            output = model(input_feature)
            total_output.append(output)
            total_label.append(label)
            # print(output,label,criterion(output, label).item())
            loss += criterion(output, label).item()

    print("Total loss:", loss)
    total_output = torch.cat(total_output, dim=0).cpu().numpy()
    total_output = total_output.reshape(-1)
    total_label = torch.cat(total_label, dim=0).cpu().numpy()
    total_label = total_label.reshape(-1)
    mae_score, rmse_score, r_2_score = evaluate(total_label, total_output)
    plt.scatter(total_label, total_output)
    plt.xlim(35,55)
    plt.ylim(35,55)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    # plt.title("pm2.5")
    plt.savefig(name + ".pdf")  # saves the plot as 'scatter_plot.png'
    plt.show()
    plt.clf()
    return mae_score, rmse_score, r_2_score


def split_data(data, row_step, col_step):
    dataset = []
    for i in range(0, data.shape[1], row_step):
        for j in range(0, data.shape[2], col_step):
            dataset.append(data[:, i : i + row_step, j : j + col_step])
    return dataset


def combine_feature(*args):
    if len(args) == 1:
        combine_feature = np.expand_dims(args[0], axis=0)
    else:
        combine_feature = np.stack(args)
    return combine_feature


def relpace_nan(data):
    # return KNNImputer().fit_transform(data)
    return IterativeImputer().fit_transform(data)
    # return SimpleImputer().fit_transform(data)


def res_parser():
    date = "2023-10-16"
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date)
    parser.add_argument("--lr", "--learning-rate", default=1e-3)
    parser.add_argument("--epoch", default=50)
    parser.add_argument("--load", default=False)
    parser.add_argument(
        "--modelurl",
        type=str,
        default="state/pre_FC.pth",
        help="model path",
    )
    parser.add_argument("--save", default=True)
    parser.add_argument("--feature", default="pm2.5")

    return parser.parse_args()


if __name__ == "__main__":
    args = res_parser()
    pm, nightlight, population, intersection_dens = load_data()
    # pm, nightlight, population = load_data()
    # nightlight.data = relpace_nan(nightlight.data)
    # population.data = relpace_nan(population.data)
    # pm.data = relpace_nan(pm.data)
    # intersection_dens.data = relpace_nan(intersection_dens.data)
    print(
        pm.data.shape,
        nightlight.data.shape,
        population.data.shape,
        intersection_dens.data.shape,
    )
    name = "FC_NY_inter"
    main(args, pm.data, [intersection_dens.data])
    name = "FC_NY_night"
    main(args, pm.data, [nightlight.data])
    name = "FC_NY_pop"
    main(args, pm.data, [population.data])
    name = "FC_NY_inter+night"
    main(args, pm.data, [intersection_dens.data,nightlight.data])
    name = "FC_NY_inter+pop"
    main(args, pm.data, [intersection_dens.data,population.data])
    name = "FC_NY_night+pop"
    main(args, pm.data, [nightlight.data,population.data])
    name = "FC_NY_inter+night+pop"
    main(args, pm.data, [intersection_dens.data,nightlight.data,population.data])
