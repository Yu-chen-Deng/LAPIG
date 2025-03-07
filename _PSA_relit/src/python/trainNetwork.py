# Author: Bingyao Huang (https://github.com/BingyaoHuang/CompenNeSt-plusplus)
# Modified by: Yuchen Deng

"""
CompenNeSt++ training functions
"""

import numpy as np
import torch

from utils import *
import ImgProc
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
import visdom

# for visualization
vis = visdom.Visdom()  # default port is 8097
assert vis.check_connection(), "Visdom: No connection, start visdom first!"

# loss functions
l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM().cuda()


# %% load training and validation data for CompenNe(S)t++
def loadData(dataset_root, data_name, cmp_name, input_size):
    # data paths
    data_root = fullfile(dataset_root, data_name)
    surface_path = fullfile(data_root, "cam/raw/ref_crop")
    cmp_train_path = fullfile(data_root, "cam/raw/cmp_crop")
    prj_train_path = fullfile(data_root, "prj/cmpwowarp")
    cmp_valid_path = fullfile(data_root, "cam/raw/cmp_crop")
    prj_valid_path = fullfile(data_root, "prj/cmpwowarp")
    print("Loading data from '{}'".format(data_root))

    # training data
    surface_d = readImgsMT(surface_path, index=[0])  # img_0001.png
    clamp_min = torch.min(surface_d).item()
    surface_l = readImgsMT(surface_path, index=[1])  # img_0125.png
    clamp_max = torch.max(surface_l).item()
    surface = readImgsMT(
        surface_path, index=[2]
    )  # ref/img_0126.png is cam-captured surface image i.e., s when img_gray.png i.e., x0 projected
    cmp_train = readImgsMT(cmp_train_path, index=list(range(1, 180)))
    prj_train = readImgsMT(prj_train_path, index=list(range(1, 180)))

    # validation data
    cmp_valid = readImgsMT(cmp_valid_path, index=list(range(180, 200)))
    prj_valid = readImgsMT(prj_valid_path, index=list(range(180, 200)))

    return surface, clamp_min, clamp_max, cmp_train, cmp_valid, prj_train, prj_valid


# initialize CompenNeSt as an autoencoder (set s=0, see journal paper sec. 3.4.3) without actual projections
def initCompenNeSt(compen_nest, dataset_root, device):
    ckpt_file = (
        "../../checkpoint/init_CompenNeSt_l1+ssim_500_24_2000_0.001_0.2_800_0.0001.pth"
    )

    if os.path.exists(ckpt_file):
        # load weights initialized CompenNet from saved state dict
        compen_nest.load_state_dict(torch.load(ckpt_file))

        print("CompenNeSt state dict found! Loading...")
    else:
        # initialize the model if checkpoint file does not exist
        print("CompenNeSt state dict not found! Initializing...")
        prj_train_path = fullfile(dataset_root, "train")

        # load data
        prj_train = readImgsMT(prj_train_path)  # x
        cam_surf = torch.zeros_like(prj_train)  # s = 0

        init_data = dict(cam_surf=cam_surf, cmp_train=prj_train, prj_train=prj_train)

        # then initialize compenNeSt as an autoencoder
        init_option = {
            "data_name": "init",
            "num_train": 500,
            "max_iters": 2000,
            "batch_size": 24,
            "lr": 1e-3,
            "lr_drop_ratio": 0.2,
            "lr_drop_rate": 800,
            "loss": "l1+ssim",
            "l2_reg": 1e-4,
            "pre-trained": False,
            "plot_on": True,
            "train_plot_rate": 100,
            "valid_rate": 200,
            "device": device,
        }

        compen_nest, _, _, _ = trainModel(compen_nest, init_data, None, init_option)

    return compen_nest


# %% train CompenNeSt_with_SL, CompenNeSt++ and CompenNeSt++FS (journal paper)
def trainModel(net, train_data, valid_data, train_option, surface_name):
    device = train_option["device"]

    # empty cuda cache before training
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # training data
    cam_surf_train = train_data["cam_surf"]
    cmp_train = train_data["cmp_train"]
    prj_train = train_data["prj_train"]

    # list of parameters to be optimized
    params = filter(
        lambda param: param.requires_grad, net.parameters()
    )  # only optimize parameters that require gradient

    # optimizer
    optimizer = optim.Adam(
        params, lr=train_option["lr"], weight_decay=train_option["l2_reg"]
    )

    # learning rate drop scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_option["lr_drop_rate"],
        gamma=train_option["lr_drop_ratio"],
    )

    # %% start train
    start_time = time.time()

    # get model name
    if not "model_name" in train_option:
        train_option["model_name"] = (
            net.name if hasattr(net, "name") else net.module.name
        )

    # initialize visdom data visualization figure
    if "plot_on" not in train_option:
        train_option["plot_on"] = True

    # title string of current training option
    title = optionToString(train_option)

    if train_option["plot_on"]:
        # intialize visdom figures
        vis_train_fig = None
        vis_valid_fig = None
        vis_curve_fig = vis.line(
            X=np.array([0]),
            Y=np.array([0]),
            name="origin",
            opts=dict(
                width=1300,
                height=500,
                markers=True,
                markersize=3,
                layoutopts=dict(
                    plotly=dict(
                        title={"text": title, "font": {"size": 18}},
                        font={"family": "Arial", "size": 20},
                        hoverlabel={"font": {"size": 20}},
                        xaxis={"title": "Iteration"},
                        yaxis={"title": "Metrics", "hoverformat": ".4f"},
                    )
                ),
            ),
        )
    # main loop
    iters = 0

    while iters < train_option["max_iters"]:
        # randomly sample training batch and send to GPU
        idx = random.sample(
            range(train_option["num_train"]), train_option["batch_size"]
        )
        cam_surf_train_batch = (
            cam_surf_train[idx, :, :, :].to(device)
            if cam_surf_train.device.type != "cuda"
            else cam_surf_train[idx, :, :, :]
        )
        cmp_train_batch = (
            cmp_train[idx, :, :, :].to(device)
            if cmp_train.device.type != "cuda"
            else cmp_train[idx, :, :, :]
        )
        prj_train_batch = (
            prj_train[idx, :, :, :].to(device)
            if prj_train.device.type != "cuda"
            else prj_train[idx, :, :, :]
        )

        # predict and compute loss
        net.train()  # explicitly set to train mode in case batchNormalization and dropout are used
        cmp_train_pred = predict(
            net, dict(prj=prj_train_batch, cam_surf=cam_surf_train_batch)
        )

        if train_option["pre-trained"]:
            # to avoid suboptimal solution (gray area), we first train with l1 loss, since ssim encourages plain gray
            if iters <= 600:
                train_loss_batch, train_l2_loss_batch = computeLoss(
                    cmp_train_pred, cmp_train_batch, "l1+l2"
                )
            else:
                train_loss_batch, train_l2_loss_batch = computeLoss(
                    cmp_train_pred, cmp_train_batch, train_option["loss"]
                )
        else:
            train_loss_batch, train_l2_loss_batch = computeLoss(
                cmp_train_pred, cmp_train_batch, train_option["loss"]
            )

        train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channel, rgb

        # backpropagation and update params
        optimizer.zero_grad()
        train_loss_batch.backward()
        optimizer.step()

        # record running time
        time_lapse = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

        # plot train
        if train_option["plot_on"]:
            if (
                iters % train_option["train_plot_rate"] == 0
                or iters == train_option["max_iters"] - 1
            ):
                vis_train_fig = plotMontage(
                    cmp_train_batch,
                    cmp_train_pred,
                    prj_train_batch,
                    win=vis_train_fig,
                    title="[Train]" + title,
                )
                appendDataPoint(
                    iters, train_loss_batch.item(), vis_curve_fig, "train_loss"
                )
                appendDataPoint(iters, train_rmse_batch, vis_curve_fig, "train_rmse")

        # validation
        valid_psnr, valid_rmse, valid_ssim = 0.0, 0.0, 0.0
        if valid_data is not None and (
            iters % train_option["valid_rate"] == 0
            or iters == train_option["max_iters"] - 1
        ):
            valid_psnr, valid_rmse, valid_ssim, cmp_valid_pred = evaluate(
                net, valid_data
            )

            # plot validation
            if train_option["plot_on"]:
                idx = np.array([1, 3, 5, 7, 8]) - 1  # fix validatio visulization
                vis_valid_fig = plotMontage(
                    valid_data["cmp_valid"][idx],
                    cmp_valid_pred[idx],
                    valid_data["prj_valid"][idx],
                    win=vis_valid_fig,
                    title="[Valid]" + title,
                )
                appendDataPoint(iters, valid_rmse, vis_curve_fig, "valid_rmse")
                appendDataPoint(iters, valid_ssim, vis_curve_fig, "valid_ssim")

        # print to console
        print(
            "Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Valid PSNR: {:7s}  | Valid RMSE: {:6s}  "
            "| Valid SSIM: {:6s}  | Learn Rate: {:.5f} |".format(
                iters,
                time_lapse,
                train_loss_batch.item(),
                train_rmse_batch,
                "{:>2.4f}".format(valid_psnr) if valid_psnr else "",
                "{:.4f}".format(valid_rmse) if valid_rmse else "",
                "{:.4f}".format(valid_ssim) if valid_ssim else "",
                optimizer.param_groups[0]["lr"],
            )
        )

        lr_scheduler.step()  # update learning rate according to schedule
        iters += 1

    # Done training and save the last epoch model
    checkpoint_dir = "./data/CMP/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file_name = fullfile(checkpoint_dir, f"{surface_name}_relit.pth")
    torch.save(net.state_dict(), checkpoint_file_name)
    print("Checkpoint saved to {}\n".format(checkpoint_file_name))

    return net, valid_psnr, valid_rmse, valid_ssim


# %% local functions


# compute loss between prediction and ground truth
def computeLoss(cmp_pred, cmp_train, loss_option):
    train_loss = 0

    # l1
    if "l1" in loss_option:
        l1_loss = l1_fun(cmp_pred, cmp_train)
        train_loss += l1_loss

    # l2
    l2_loss = l2_fun(cmp_pred, cmp_train)
    if "l2" in loss_option:
        train_loss += l2_loss

    # ssim
    if "ssim" in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(cmp_pred, cmp_train))
        train_loss += ssim_loss

    return train_loss, l2_loss


# append a data point to the curve in win
def appendDataPoint(x, y, win, name, env=None):
    vis.line(
        X=np.array([x]),
        Y=np.array([y]),
        env=env,
        win=win,
        update="append",
        name=name,
        opts=dict(markers=True, markersize=3),
    )


# plot sample predicted images using visdom, more than three rows
def plotMontage(*argv, index=None, win=None, title=None, env=None):
    with torch.no_grad():  # just in case
        # compute montage grid size
        if argv[0].shape[0] > 5:
            grid_w = 5
            idx = (
                random.sample(range(argv[0].shape[0]), grid_w)
                if index is None
                else index
            )
        else:
            grid_w = argv[0].shape[0]
            # idx = random.sample(range(cam_im.shape[0]), grid_w)
            idx = range(grid_w)

        # resize to (256, 256) for better display
        tile_size = (256, 256)
        im_resize = torch.empty((len(argv) * grid_w, argv[0].shape[1]) + tile_size)
        i = 0
        for im in argv:
            if im.shape[2] != tile_size[0] or im.shape[3] != tile_size[1]:
                im_resize[i : i + grid_w] = F.interpolate(im[idx, :, :, :], tile_size)
            else:
                im_resize[i : i + grid_w] = im[idx, :, :, :]
            i += grid_w

        # title
        plot_opts = dict(
            title=title,
            caption=title,
            font=dict(size=18),
            width=1300,
            store_history=False,
        )

        im_montage = torchvision.utils.make_grid(
            im_resize, nrow=grid_w, padding=10, pad_value=1
        )
        win = vis.image(im_montage, win=win, opts=plot_opts, env=env)

    return win


# predict projector input images given input data (do not use with torch.no_grad() within this function)
def predict(net, data):
    if "cam_surf" in data and data["cam_surf"] is not None:
        cmp_pred = net(data["prj"], data["cam_surf"])
    else:
        cmp_pred = net(data["prj"])

    if type(cmp_pred) == tuple and len(cmp_pred) > 1:
        cmp_pred = cmp_pred[0]
    return cmp_pred


# evaluate model on validation dataset
def evaluate(net, valid_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cam_surf = valid_data["cam_surf"]
    cmp_valid = valid_data["cmp_valid"]
    prj_valid = valid_data["prj_valid"]

    with torch.no_grad():
        net.eval()  # explicitly set to eval mode

        # if you have limited GPU memory, we need to predict in batch mode
        if cam_surf.device.type != device.type:
            last_loc = 0
            valid_mse, valid_ssim = 0.0, 0.0

            cmp_valid_pred = torch.zeros(cmp_valid.shape)
            num_valid = cmp_valid.shape[0]
            batch_size = (
                50 if num_valid > 50 else num_valid
            )  # default number of test images per dataset

            for i in range(0, num_valid // batch_size):
                idx = range(last_loc, last_loc + batch_size)
                cam_surf_batch = (
                    cam_surf[idx, :, :, :].to(device)
                    if cam_surf.device.type != "cuda"
                    else cam_surf[idx, :, :, :]
                )
                cmp_valid_batch = (
                    cmp_valid[idx, :, :, :].to(device)
                    if cmp_valid.device.type != "cuda"
                    else cmp_valid[idx, :, :, :]
                )
                prj_valid_batch = (
                    prj_valid[idx, :, :, :].to(device)
                    if prj_valid.device.type != "cuda"
                    else prj_valid[idx, :, :, :]
                )

                # predict batch
                cmp_valid_pred_batch = predict(
                    net, dict(prj=prj_valid_batch, cam_surf=cam_surf_batch)
                ).detach()
                if (
                    type(cmp_valid_pred_batch) == tuple
                    and len(cmp_valid_pred_batch) > 1
                ):
                    cmp_valid_pred_batch = cmp_valid_pred_batch[0]
                cmp_valid_pred[last_loc : last_loc + batch_size, :, :, :] = (
                    cmp_valid_pred_batch.cpu()
                )

                # compute loss
                valid_mse += (
                    l2_fun(cmp_valid_pred_batch, cmp_valid_batch).item() * batch_size
                )
                valid_ssim += ssim(cmp_valid_pred_batch, cmp_valid_batch) * batch_size

                last_loc += batch_size
            # average
            valid_mse /= num_valid
            valid_ssim /= num_valid

            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
        else:
            # if all data can be loaded to GPU memory
            cmp_valid_pred = predict(
                net, dict(prj=prj_valid, cam_surf=cam_surf)
            ).detach()
            valid_mse = l2_fun(cmp_valid_pred, cmp_valid).item()
            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
            valid_ssim = ssim_fun(cmp_valid_pred, cmp_valid).item()

    return valid_psnr, valid_rmse, valid_ssim, cmp_valid_pred
