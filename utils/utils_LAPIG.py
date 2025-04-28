# Partially Obtained from https://github.com/BingyaoHuang/CompenNeSt-plusplus (Author: Bingyao Huang)

import socket
import subprocess
import yaml
import os
from os.path import join

import time
import sys

import cv2 as cv
import torch
import numpy as np
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import platform
from CompenNeStPlusplusDataset import SimpleDataset


def load_config(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None


def get_config(config, *keys):
    for key in keys:
        if isinstance(config, dict):
            config = config.get(key)
        else:
            return None
    return config


def check_and_start_visdom(port=8097, timeout=10):
    def is_port_in_use(check_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", check_port)) == 0

    def start_visdom_process():
        startupinfo = None
        creationflags = 0
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        else:
            pass

        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "visdom.server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                creationflags=creationflags,
                close_fds=True,
            )
            return process
        except Exception as e:
            print(f"Error starting Visdom: {str(e)}")
            return None

    if is_port_in_use(port):
        print("✅ Visdom is already running")
        return True

    print("🔄 Trying to start Visdom...")
    process = start_visdom_process()

    if not process:
        print("❌ Automatically starting Visdom failed")
        manual_start_prompt()
        return False

    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            print("✅ Visdom started successfully")
            return True
        time.sleep(1)

    process.terminate()
    print("❌ Visdom startup timeout")
    print(f"Please execute the command manually to start Visdom：\n{sys.executable} -m visdom.server")
    manual_start_prompt()
    return False


def manual_start_prompt():
    try:
        input("🛑 Please start Visdom manually and press Enter to continue...")
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        exit(1)


def execute(script_path, wait=True, args=None):
    if args is None:
        args = []
    cmd = ["python", script_path] + args
    print(f"Starting: {' '.join(cmd)}")

    process = subprocess.Popen(cmd)

    if wait:
        process.wait()
        if process.returncode != 0:
            print(f"Exited with error: {script_path}")
            sys.exit(1)
    else:
        time.sleep(0.1)
        if process.poll() is not None and process.returncode != 0:
            print(f"Failed to start: {script_path}")
            sys.exit(1)

    return process


def PIL2Tensor4D(PIL_img, device="cpu"):
    transform = transforms.ToTensor()  # (H, W, C) -> (C, H, W)

    # (3, 256, 256) -> (1, 3, 256, 256)
    Tensor_img = transform(PIL_img).unsqueeze(0).to(device)

    return Tensor_img


def Tensor4D2PIL(Tensor_img):
    transform = transforms.ToPILImage()  # (C, H, W) -> PIL image

    # (1, 3, H, W) -> (3, H, W)
    Tensor_img = Tensor_img.squeeze(0)

    PIL_img = transform(Tensor_img)

    return PIL_img


def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# read images using multi-thread
def readImgsMT(img_dir, size=None, index=None, gray_scale=False, normalize=False):
    img_dataset = SimpleDataset(img_dir, index=index, size=size)
    if platform.system() == "Windows":
        data_loader = DataLoader(
            img_dataset,
            batch_size=len(img_dataset),
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
    else:
        data_loader = DataLoader(
            img_dataset,
            batch_size=len(img_dataset),
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )

    for i, imgs in enumerate(data_loader):
        # imgs.permute((0, 3, 1, 2)).to('cpu', dtype=torch.float32)/255
        # convert to torch.Tensor
        imgs = imgs.permute((0, 3, 1, 2)).float().div(255)

        if gray_scale:
            imgs = (
                0.2989 * imgs[:, 0] + 0.5870 * imgs[:, 1] + 0.1140 * imgs[:, 2]
            )  # same as MATLAB rgb2gray and OpenCV cvtColor
            imgs = imgs[:, None]

        # normalize to [-1, 1], should improve model convergence in early training stages.
        if normalize:
            imgs = (imgs - 0.5) / 0.5

        return imgs


# save 4D np.ndarray or torch tensor to image files
def saveImgs(inputData, dir, name=None):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if type(inputData) is torch.Tensor:
        if inputData.requires_grad:
            inputData = inputData.detach()
        if inputData.device.type == "cuda":
            imgs = (
                inputData.cpu().numpy().transpose(0, 2, 3, 1)
            )  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(
                0, 2, 3, 1
            )  # (N, C, row, col) to (N, row, col, C)

    else:
        imgs = inputData

    # imgs must have a shape of (N, row, col, C)
    imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv
    for i in range(imgs.shape[0]):
        if i is 0 and name is not None:
            file_name = name
        else:
            file_name = "img_{:04d}.png".format(i + 1)
        cv.imwrite(join(dir, file_name), imgs[i, :, :, :])  # faster than PIL or scipy
