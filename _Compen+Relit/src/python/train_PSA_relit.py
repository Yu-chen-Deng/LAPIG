# Author: Bingyao Huang (https://github.com/BingyaoHuang/CompenNeSt-plusplus)
# Modified by: Yuchen Deng

"""
Training and testing script for CompenNeSt++ (journal extension of cvpr'19 and iccv'19 papers)

This script trains/tests CompenNeSt++ on different dataset specified in 'data_list' below.
The detailed training options are given in 'train_option' below.

1. We start by setting the training environment to GPU (if any).
2. K=20 setups are listed in 'data_list', which are our full compensation benchmark.
3. We set number of training images to 500 and loss function to l1+ssim, you can add other num_train and loss to 'num_train_list' and 'loss_list' for
comparison. Other training options are specified in 'train_option'.
4. The training data 'train_data' and validation data 'valid_data', are loaded in RAM using function 'loadData', and then we train the model with
function 'trainCompenNeStModel'. The training and validation results are both updated in Visdom window (`http://server:8098`) and console.
5. Once the training is finished, we can compensate the desired image. The compensation images 'prj_cmp_test' can then be projected to the surface.

Example:
    python train_compenNeSt++.py

See Models.py for CompenNeSt++ structure.
See trainNetwork.py for detailed training process.
See utils.py for helper functions.

Citation:
    @article{huang2020CompenNeSt++,
        title={End-to-end Full Projector Compensation},
        author={Bingyao Huang and Tao Sun and Haibin Ling},
        year={2021},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)} }

    @inproceedings{huang2019compennet++,
        author = {Huang, Bingyao and Ling, Haibin},
        title = {CompenNet++: End-to-end Full Projector Compensation},
        booktitle = {IEEE International Conference on Computer Vision (ICCV)},
        month = {October},
        year = {2019} }

    @inproceedings{huang2019compennet,
        author = {Huang, Bingyao and Ling, Haibin},
        title = {End-To-End Projector Photometric Compensation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019} }
"""

# %% Set environment
import os

# set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

from trainRelitNetwork import *
import RelitModels
import csv
import yaml


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() >= 1:
    print("Train with", len(device_ids), "GPUs!")
else:
    print("Train with CPU!")

config_path = "config.yaml"
config = load_yaml(config_path)
surface_name = config.get("setup_name")
surface = config.get("setup_name")

# %% K=20 setups
dataset_root = fullfile("./data/Compen+Relit")

data_list = [surface]

cmp_list = ["Compensation"]

# Training configurations of CompenNet++ reported in the paper
num_train_list = [125]
loss_list = ["l1+l2+ssim"]
fine_tune = False
# ckpt_file = './_Relit/checkpoint/blender_APRIF_PCNet_clamp_CmpNeStpp_l1+l2+ssim_10000_32_30000_0.0015_0.8_2000_0.0001.pth'
ckpt_file = "./_Relit/checkpoint/_vset01_APRIF_PCNet_clamp_CmpNeStpp_l1+l2+ssim_1250_32_10000_0.0015_0.8_2000_0.0001.pth"

# You can also compare different configurations, such as different number of training images and loss functions as shown below
# num_train_list = [8, 48, 125, 250, 500, 1000]
# loss_list = ['l1', 'l2', 'ssim', 'l1+l2', 'l1+ssim', 'l2+ssim', 'l1+l2+ssim']
# loss_list = ['l1', 'l2', 'l1+l2']

# You can create your own models in Models.py and put their names in this list for comparisons.
# model_list = ['PSA_relit_PCNet', 'PSA_relit_PCNet', 'PSA_relit_CompenNeStPlusplus_clamp', 'PSA_relit_CompenNeStPlusplus']

model_list = ["PSA_relit_PCNet"]

# default training options
train_option_default = {
    "max_iters": 1500,
    "batch_size": 8,
    "lr": 1e-3,  # learning rate
    "lr_drop_ratio": 0.2,
    "lr_drop_rate": 1000,  # adjust this according to max_iters (lr_drop_rate < max_iters)
    "loss": "",  # loss will be set to one of the loss functions in loss_list later
    "l2_reg": 1e-4,  # l2 regularization
    "device": device,
    "pre-trained": True,
    "plot_on": True,  # plot training progress using visdom (disable for faster training)
    "train_plot_rate": 100,  # training and visdom plot rate (increase for faster training)
    "valid_rate": 100,
}  # validation and visdom plot rate (increase for faster training)

# a flag that decides whether to compute and save the compensated images to the drive
save_test = False

# log file
from time import localtime, strftime

log_dir = "../../log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_name = strftime("%Y-%m-%d_%H_%M_%S", localtime()) + ".csv"
fieldnames = [
    "data_name",
    "model_name",
    "cmp_name",
    "loss_function",
    "num_train",
    "batch_size",
    "max_iters",
    "uncmp_psnr",
    "uncmp_rmse",
    "uncmp_ssim",
    "valid_psnr",
    "valid_rmse",
    "valid_ssim",
]

with open(fullfile(log_dir, log_file_name), mode="w", newline="") as log_file:
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

# resize the input images if input_size is not None
input_size = None
# input_size = (256, 256) # we can also use a low-res input to reduce memory usage and speed up training/testing with a sacrifice of precision
resetRNGseed(0)


def main():
    # stats for different setups
    for data_name in data_list:
        for cmp_name in cmp_list:
            train_option = train_option_default.copy()
            train_option["cmp_name"] = cmp_name.replace("/", "_")

            # load training and validation data
            data_root = fullfile(dataset_root, data_name)

            (
                surface,
                clamp_min,
                clamp_max,
                cmp_train,
                cmp_valid,
                prj_train,
                prj_valid,
            ) = loadData(dataset_root, data_name, cmp_name, input_size)

            # surface image for training and validation
            cmp_surf_train = surface.expand_as(cmp_train)
            cmp_surf_valid = surface.expand_as(cmp_valid)

            # convert valid data to CUDA tensor if you have sufficient GPU memory (significant speedup)
            cmp_valid = cmp_valid.to(device)
            prj_valid = prj_valid.to(device)

            # validation data, 200 image pairs
            valid_data = dict(
                cam_surf=cmp_surf_valid, cmp_valid=cmp_valid, prj_valid=prj_valid
            )

            # stats for different #Train
            for num_train in num_train_list:
                train_option["num_train"] = num_train

                # select a subset to train
                train_data = dict(
                    cam_surf=cmp_surf_train[:num_train, :, :, :],
                    cmp_train=cmp_train[:num_train, :, :, :],
                    prj_train=prj_train[:num_train, :, :, :],
                )

                # stats for different models
                for model_name in model_list:
                    # create PSA_relit
                    if model_name == (
                        "PSA_relit_PCNet" or "PSA_relit_CompenNeStPlusplus_clamp"
                    ):
                        PSA_relit = getattr(RelitModels, model_name)(
                            clamp_min=clamp_min, clamp_max=clamp_max
                        )
                    else:
                        PSA_relit = getattr(RelitModels, model_name)()

                    if torch.cuda.device_count() >= 1:
                        PSA_relit = nn.DataParallel(
                            PSA_relit, device_ids=device_ids
                        ).to(device)

                    if fine_tune:
                        PSA_relit.load_state_dict(torch.load(ckpt_file))

                    train_option["model_name"] = model_name.replace("/", "_")

                    # stats for different loss functions
                    for loss in loss_list:
                        # set seed of rng for repeatability
                        resetRNGseed(0)

                        # train option for current configuration, i.e., data name and loss function
                        train_option["data_name"] = data_name.replace("/", "_")
                        train_option["loss"] = loss

                        print(
                            "-------------------------------------- Training Options -----------------------------------"
                        )
                        print(
                            "\n".join(
                                "{}: {}".format(k, v) for k, v in train_option.items()
                            )
                        )
                        print(
                            "------------------------------------ Start training {:s} ---------------------------".format(
                                model_name
                            )
                        )

                        # train model
                        PSA_relit, valid_psnr, valid_rmse, valid_ssim = trainModel(
                            PSA_relit,
                            train_data,
                            valid_data,
                            train_option,
                            surface_name,
                        )

                        # save results to log file
                        data_row = {
                            "data_name": data_name,
                            "model_name": model_name,
                            "cmp_name": cmp_name,
                            "loss_function": loss,
                            "num_train": num_train,
                            "batch_size": train_option["batch_size"],
                            "max_iters": train_option["max_iters"],
                            "uncmp_psnr": 0,
                            "uncmp_rmse": 0,
                            "uncmp_ssim": 0,
                            "valid_psnr": valid_psnr,
                            "valid_rmse": valid_rmse,
                            "valid_ssim": valid_ssim,
                        }

                        with open(
                            fullfile(log_dir, log_file_name), mode="a", newline=""
                        ) as log_file:
                            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
                            writer.writerow(data_row)

                        # [testing phase] create compensated testing images
                        if save_test:
                            print(
                                "------------------------------------ Start testing {:s} ---------------------------".format(
                                    model_name
                                )
                            )
                            torch.cuda.empty_cache()

                            # desired test images are created such that they can fill the optimal displayable area (see paper for detail)
                            desire_test_path = fullfile(data_root, "cam/desire/test")
                            assert os.path.isdir(
                                desire_test_path
                            ), "images and folder {:s} does not exist!".format(
                                desire_test_path
                            )

                            # compensate and save images
                            projection_test = readImgsMT(
                                desire_test_path, index=list(range(1000, 1200))
                            ).to(device)
                            surface_test = surface.expand_as(projection_test).to(device)
                            with torch.no_grad():
                                # compensate using CompenNet++
                                PSA_relit.eval()
                                prj_cmp_test = PSA_relit(
                                    projection_test, surface_test
                                ).detach()  # compensated prj input image x^{*}
                            del projection_test, surface_test

                            # create image save path
                            cmp_folder_name = "{}_{}_{}_{}_{}_{}".format(
                                train_option["model_name"],
                                train_option["cmp_name"],
                                loss,
                                num_train,
                                train_option["batch_size"],
                                train_option["max_iters"],
                            )
                            prj_cmp_path = fullfile(
                                data_root, "result", cmp_folder_name
                            )
                            if not os.path.exists(prj_cmp_path):
                                os.makedirs(prj_cmp_path)

                            # save images
                            saveImgs(
                                prj_cmp_test, prj_cmp_path
                            )  # compensated testing images, i.e., to be projected to the surface
                            print("Compensation images saved to " + prj_cmp_path)

                        print(
                            "-------------------------------------- Done! ---------------------------\n"
                        )
                    # clear cache
                    del PSA_relit
                    torch.cuda.empty_cache()
                del train_data
            del cmp_valid, prj_valid

    print("All dataset done!")


if __name__ == "__main__":
    main()
