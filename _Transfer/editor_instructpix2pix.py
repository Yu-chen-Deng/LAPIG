# Obtained from https://github.com/volotat/DiffMorph (Author: Alexey Borsky)
# Modified by: Yuchen Deng

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from os.path import join

# Add the path to your module to the system path
module_path = os.path.abspath(join(".", "_Compensation", "src", "python"))
sys.path.append(module_path)

from tqdm import tqdm
import time
import yaml
from transformers import CLIPProcessor, CLIPModel
import PIL
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms.functional as TF
import random
import tensorflow as tf
import cv2
import subprocess
import Models
from trainNetwork import *
from utils import readImgsMT, saveImgs
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
import numpy as np
import multiprocessing as mp
from multiprocessing import Array
import argparse
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_tps

# Define the perspective transformation matrix (you need to replace it with your actual matrix)
M = np.array(
    [
        [1.48313571e00, -7.13808630e-02, -2.36984465e01],
        [1.41877995e-01, 1.37148728e00, -6.01562699e01],
        [8.49259546e-04, -5.31231758e-05, 1.00000000e00],
    ],
    dtype=np.float32,
)  # 3x3 Matrix

# Morph image size
im_sz = 256
# Morph mapping size
mp_sz = 96


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None


def start_process(script_name):
    try:
        process = subprocess.Popen(["python", script_name])
        time.sleep(0.1)  # Waiting for the process to start
        if process.poll() is not None:
            print(f"Failed to start {script_name}")
            sys.exit(0.1)
        return process
    except Exception as e:
        print(f"Error starting {script_name}: {e}")
        sys.exit(0.1)


def dense_image_warp(image, flow):
    batch_size, height, width, channels = image.shape
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    grid = tf.stack([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, tf.float32)
    flow = tf.reshape(flow, [batch_size, height, width, 2])
    new_locations = grid + flow
    new_locations = tf.reshape(new_locations, [batch_size, height, width, 2])

    new_x = new_locations[..., 0]
    new_y = new_locations[..., 1]

    new_x = tf.clip_by_value(new_x, 0, width - 1)
    new_y = tf.clip_by_value(new_y, 0, height - 1)

    x0 = tf.cast(tf.floor(new_x), tf.int32)
    x1 = tf.cast(tf.floor(new_x + 1), tf.int32)
    y0 = tf.cast(tf.floor(new_y), tf.int32)
    y1 = tf.cast(tf.floor(new_y + 1), tf.int32)

    x0 = tf.clip_by_value(x0, 0, width - 1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y0 = tf.clip_by_value(y0, 0, height - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)

    Ia = tf.gather_nd(image, tf.stack([y0, x0], axis=-1), batch_dims=1)
    Ib = tf.gather_nd(image, tf.stack([y1, x0], axis=-1), batch_dims=1)
    Ic = tf.gather_nd(image, tf.stack([y0, x1], axis=-1), batch_dims=1)
    Id = tf.gather_nd(image, tf.stack([y1, x1], axis=-1), batch_dims=1)

    wa = (tf.cast(x1, tf.float32) - new_x) * (tf.cast(y1, tf.float32) - new_y)
    wb = (tf.cast(x1, tf.float32) - new_x) * (new_y - tf.cast(y0, tf.float32))
    wc = (new_x - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - new_y)
    wd = (new_x - tf.cast(x0, tf.float32)) * (new_y - tf.cast(y0, tf.float32))

    wa = tf.expand_dims(wa, axis=-1)
    wb = tf.expand_dims(wb, axis=-1)
    wc = tf.expand_dims(wc, axis=-1)
    wd = tf.expand_dims(wd, axis=-1)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


# warp function
@tf.function
def warp(origins, targets, preds_org, preds_trg):
    if add_first:
        res_targets = dense_image_warp(
            image=(origins + preds_org[:, :, :, 3:6] * 2 * add_scale)
            * tf.maximum(0.1, 1 + preds_org[:, :, :, 0:3] * mult_scale),
            flow=preds_org[:, :, :, 6:8] * im_sz * warp_scale,
        )
        res_origins = dense_image_warp(
            image=(targets + preds_trg[:, :, :, 3:6] * 2 * add_scale)
            * tf.maximum(0.1, 1 + preds_trg[:, :, :, 0:3] * mult_scale),
            flow=preds_trg[:, :, :, 6:8] * im_sz * warp_scale,
        )
    else:
        res_targets = dense_image_warp(
            image=origins * tf.maximum(0.1, 1 + preds_org[:, :, :, 0:3] * mult_scale)
            + preds_org[:, :, :, 3:6] * 2 * add_scale,
            flow=preds_org[:, :, :, 6:8] * im_sz * warp_scale,
        )
        res_origins = dense_image_warp(
            image=targets * tf.maximum(0.1, 1 + preds_trg[:, :, :, 0:3] * mult_scale)
            + preds_trg[:, :, :, 3:6] * 2 * add_scale,
            flow=preds_trg[:, :, :, 6:8] * im_sz * warp_scale,
        )
    return res_targets, res_origins


# create morphing grid
def create_grid(scale):
    grid = np.mgrid[0:scale, 0:scale] / (scale - 1) * 2 - 1
    grid = np.swapaxes(grid, 0, 2)
    grid = np.expand_dims(grid, axis=0)
    return grid


# produce morphing warp maps
def produce_warp_maps(origins, targets):
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(64, (5, 5))
            self.act1 = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.conv2 = tf.keras.layers.Conv2D(64, (5, 5))
            self.act2 = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.convo = tf.keras.layers.Conv2D((3 + 3 + 2) * 2, (5, 5))

        def call(self, maps):
            x = tf.image.resize(maps, [mp_sz, mp_sz])
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.convo(x)
            return x

    model = MyModel()

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    train_loss = tf.keras.metrics.Mean(name="train_loss")

    @tf.function
    def train_step(maps, origins, targets):
        with tf.GradientTape() as tape:
            preds = model(maps)
            preds = tf.image.resize(preds, [im_sz, im_sz])

            res_targets_, res_origins_ = warp(
                origins, targets, preds[..., :8], preds[..., 8:]
            )

            res_map = dense_image_warp(
                image=maps, flow=preds[:, :, :, 6:8] * im_sz * warp_scale
            )
            res_map = dense_image_warp(
                image=res_map, flow=preds[:, :, :, 14:16] * im_sz * warp_scale
            )

            loss = (
                loss_object(maps, res_map) * 1
                + loss_object(res_targets_, targets) * 0.3
                + loss_object(res_origins_, origins) * 0.3
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    maps = create_grid(im_sz)
    maps = np.concatenate((maps, origins * 0.1, targets * 0.1), axis=-1).astype(
        np.float32
    )

    for i in tqdm(range(TRAIN_EPOCHS), desc="Morph Training", unit="it"):
        epoch = i + 1

        train_step(maps, origins, targets)

        if (
            (epoch < 100 and epoch % 10 == 0)
            or (epoch < 1000 and epoch % 100 == 0)
            or (epoch % 1000 == 0)
        ):
            preds = model(maps, training=False)[:1]
            preds = tf.image.resize(preds, [im_sz, im_sz])

            res_targets, res_origins = warp(
                origins, targets, preds[..., :8], preds[..., 8:]
            )

            res_targets = tf.clip_by_value(res_targets, -1, 1)[0]
            res_img = ((res_targets.numpy() + 1) * 127.5).astype(np.uint8)

            res_img = cv2.resize(
                res_img, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_AREA
            )

            cv2.imwrite(
                "./_Morph/train/a_to_b_%d.jpg" % epoch,
                cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR),
            )

            res_origins = tf.clip_by_value(res_origins, -1, 1)[0]
            res_img = ((res_origins.numpy() + 1) * 127.5).astype(np.uint8)

            res_img = cv2.resize(
                res_img, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_AREA
            )

            cv2.imwrite(
                "./_Morph/train/b_to_a_%d.jpg" % epoch,
                cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR),
            )

            # Save the weights file to disk
            # np.save('./_Morph/preds.npy', preds.numpy())
            # Save the weights file to memory
            preds_np = preds.numpy().flatten()

    shared_array = mp.Array("f", preds_np)

    return shared_array


# generating frames using deformation mapping
def use_warp_maps(origins, targets, steps, preds_shared_memory):
    STEPS = steps

    # Load the weights file from disk
    # preds = np.load('./_Morph/preds.npy')
    # Load the weights file from memory
    preds_np = np.frombuffer(preds_shared_memory.get_obj(), dtype=np.float32)
    preds = preds_np.reshape((1, im_sz, im_sz, 16))

    # save the map as an image
    res_img = np.zeros((im_sz * 2, im_sz * 3, 3))

    res_img[im_sz * 0 : im_sz * 1, im_sz * 0 : im_sz * 1] = preds[
        0, :, :, 0:3
    ]  # a_to_b addition mapping
    res_img[im_sz * 0 : im_sz * 1, im_sz * 1 : im_sz * 2] = preds[
        0, :, :, 3:6
    ]  # a_to_b multiplication mapping
    res_img[im_sz * 0 : im_sz * 1, im_sz * 2 : im_sz * 3, :2] = preds[
        0, :, :, 6:8
    ]  # a_to_b deformation mapping

    res_img[im_sz * 1 : im_sz * 2, im_sz * 0 : im_sz * 1] = preds[
        0, :, :, 8:11
    ]  # b_to_a addition mapping
    res_img[im_sz * 1 : im_sz * 2, im_sz * 1 : im_sz * 2] = preds[
        0, :, :, 11:14
    ]  # b_to_a multiplication mapping
    res_img[im_sz * 1 : im_sz * 2, im_sz * 2 : im_sz * 3, :2] = preds[
        0, :, :, 14:16
    ]  # b_to_a deformation mapping

    res_img = np.clip(res_img, -1, 1)
    res_img = ((res_img + 1) * 127.5).astype(np.uint8)
    cv2.imwrite("./_Morph/morph/maps.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

    # apply the mapping and save the result

    org_strength = tf.reshape(tf.range(STEPS, dtype=tf.float32), [STEPS, 1, 1, 1]) / (
        STEPS - 1
    )
    trg_strength = tf.reverse(org_strength, axis=[0])

    res_img = np.zeros((im_sz * 3, im_sz * (STEPS // 10), 3), dtype=np.uint8)

    for i in tqdm(range(STEPS), desc="Morph Transing", unit="it"):
        preds_org = preds * org_strength[i]
        preds_trg = preds * trg_strength[i]

        res_targets, res_origins = warp(
            origins, targets, preds_org[..., :8], preds_trg[..., 8:]
        )
        res_targets = tf.clip_by_value(res_targets, -1, 1)
        res_origins = tf.clip_by_value(res_origins, -1, 1)

        results = res_targets * trg_strength[i] + res_origins * org_strength[i]
        res_numpy = results.numpy()

        img = ((res_numpy[0] + 1) * 127.5).astype(np.uint8)
        output_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        output_img = cv2.resize(
            output_img, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_AREA
        )

        # save the uncompensated image to disk
        # cv2.imwrite(join("./data/NST", surface, 'uncmp', 'img_{:04d}.png'.format(i+1)), output_img)

        # save the uncompensated image to memory
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        np_shared_array[i] = output_img

        if (i + 1) % 10 == 0:
            res_img[im_sz * 0 : im_sz * 1, i // 10 * im_sz : (i // 10 + 1) * im_sz] = (
                img
            )
            res_img[im_sz * 1 : im_sz * 2, i // 10 * im_sz : (i // 10 + 1) * im_sz] = (
                (res_targets.numpy()[0] + 1) * 127.5
            ).astype(np.uint8)
            res_img[im_sz * 2 : im_sz * 3, i // 10 * im_sz : (i // 10 + 1) * im_sz] = (
                (res_origins.numpy()[0] + 1) * 127.5
            ).astype(np.uint8)

    cv2.imwrite("./_Morph/morph/result.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()


class SurfaceDataset(Dataset):
    def __init__(self, surface_dir, projection_dir, capture_dir):
        self.surface_images = [
            os.path.join(surface_dir, f)
            for f in os.listdir(surface_dir)
            if f.endswith(".png")
        ]
        self.projection_images = [
            os.path.join(projection_dir, f)
            for f in os.listdir(projection_dir)
            if f.endswith(".png")
        ]
        self.capture_images = [
            [
                os.path.join(capture_dir, surface, f)
                for f in os.listdir(os.path.join(capture_dir, surface))
                if f.endswith(".png")
            ]
            for surface in os.listdir(capture_dir)
        ]

        self.total_projections = len(self.projection_images)

    def __len__(self):
        return len(self.surface_images) * self.total_projections

    def __getitem__(self, idx):
        surface_idx = idx // self.total_projections
        proj_idx = idx % self.total_projections

        try:
            surface_image = cv2.imread(
                self.surface_images[surface_idx], cv2.IMREAD_COLOR
            )
            projection_image = cv2.imread(
                self.projection_images[proj_idx], cv2.IMREAD_COLOR
            )
            capture_image = cv2.imread(
                self.capture_images[surface_idx][proj_idx], cv2.IMREAD_COLOR
            )

            # Preprocessing: resizing and normalization
            surface_image = cv2.resize(surface_image, (256, 256)) / 255.0
            projection_image = cv2.resize(projection_image, (256, 256)) / 255.0
            capture_image = cv2.resize(capture_image, (256, 256)) / 255.0

            # Combine the surface image and the projection image as input, concatenate the channels, and make sure input_image is (6, 256, 256)
            input_image = np.concatenate(
                (surface_image, projection_image), axis=2
            )  # Combine color channels
            input_image = np.transpose(
                input_image, (2, 0, 1)
            )  # Convert to (6, 256, 256)

            # Process the target and convert it into 3 channels
            capture_image = np.transpose(
                capture_image, (2, 0, 1)
            )  # Convert to (3, 256, 256)

            return (
                torch.tensor(input_image, dtype=torch.float32),
                torch.tensor(capture_image, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Error loading image: {e}")
            return self.__getitem__(
                (idx + 1) % len(self)
            )  # Skip this image and try the next one


class WarpingNet(nn.Module):
    def __init__(self, grid_shape=(6, 6), out_size=(256, 256), with_refine=True):
        super(WarpingNet, self).__init__()
        self.grid_shape = grid_shape
        self.out_size = tuple(out_size)
        self.with_refine = with_refine  # becomes WarpingNet w/o refine if set to false
        self.name = (
            self.__class__.__name__
            if with_refine
            else self.__class__.__name__ + "_without_refine"
        )

        # relu
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        # final refined grid
        self.register_buffer("fine_grid", None)

        # affine params
        self.affine_mat = nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]).view(-1, 2, 3))

        # tps params
        self.nctrl = self.grid_shape[0] * self.grid_shape[1]
        self.nparam = self.nctrl + 2
        ctrl_pts = pytorch_tps.uniform_grid(grid_shape)
        self.register_buffer("ctrl_pts", ctrl_pts.view(-1, 2))
        self.theta = nn.Parameter(
            torch.ones((1, self.nparam * 2), dtype=torch.float32).view(
                -1, self.nparam, 2
            )
            * 1e-3
        )

        # initialization function, first checks the module type,
        def init_normal(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)

        # grid refinement net
        if self.with_refine:
            self.grid_refine_net = nn.Sequential(
                nn.Conv2d(2, 32, 3, 2, 1),
                self.relu,
                nn.Conv2d(32, 64, 3, 2, 1),
                self.relu,
                nn.ConvTranspose2d(64, 32, 2, 2, 0),
                self.relu,
                nn.ConvTranspose2d(32, 2, 2, 2, 0),
                self.leakyRelu,
            )
            self.grid_refine_net.apply(init_normal)
        else:
            self.grid_refine_net = None  # WarpingNet w/o refine

    # initialize WarpingNet's affine matrix to the input affine_vec
    def set_affine(self, affine_vec):
        self.affine_mat.data = torch.Tensor(affine_vec).view(-1, 2, 3)

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, x):
        # generate coarse affine and TPS grids
        coarse_affine_grid = F.affine_grid(
            self.affine_mat,
            torch.Size([1, x.shape[1], x.shape[2], x.shape[3]]),
            align_corners=True,
        ).permute((0, 3, 1, 2))
        coarse_tps_grid = pytorch_tps.tps_grid(
            self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size
        )

        # use TPS grid to sample affine grid
        tps_grid = F.grid_sample(
            coarse_affine_grid, coarse_tps_grid, align_corners=True
        )

        # refine TPS grid using grid refinement net and save it to self.fine_grid
        if self.with_refine:
            self.fine_grid = torch.clamp(
                self.grid_refine_net(tps_grid) + tps_grid, min=-1, max=1
            ).permute((0, 2, 3, 1))
        else:
            self.fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))

    def forward(self, x):

        if self.fine_grid is None:
            # not simplified (training/validation)
            # generate coarse affine and TPS grids
            coarse_affine_grid = F.affine_grid(
                self.affine_mat,
                torch.Size([1, x.shape[1], x.shape[2], x.shape[3]]),
                align_corners=True,
            ).permute((0, 3, 1, 2))
            coarse_tps_grid = pytorch_tps.tps_grid(
                self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size
            )

            # use TPS grid to sample affine grid
            tps_grid = F.grid_sample(
                coarse_affine_grid, coarse_tps_grid, align_corners=True
            ).repeat(x.shape[0], 1, 1, 1)

            # refine TPS grid using grid refinement net and save it to self.fine_grid
            if self.with_refine:
                fine_grid = torch.clamp(
                    self.grid_refine_net(tps_grid) + tps_grid, min=-1, max=1
                ).permute((0, 2, 3, 1))
            else:
                fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            # simplified (testing)
            fine_grid = self.fine_grid.repeat(x.shape[0], 1, 1, 1)

        # warp
        x = F.grid_sample(x, fine_grid, align_corners=True)
        return x


class ShadingNetSPAA_clamp(nn.Module):
    # Extended from CompenNet
    def __init__(self, use_rough=True):
        super(ShadingNetSPAA_clamp, self).__init__()
        self.use_rough = use_rough
        self.name = (
            self.__class__.__name__
            if self.use_rough
            else self.__class__.__name__ + "_no_rough"
        )
        self.relu = nn.ReLU()

        # backbone branch
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # surface image feature extraction branch
        num_chan = 6 if self.use_rough else 3
        self.conv1_s = nn.Conv2d(num_chan, 32, 3, 2, 1)
        self.conv2_s = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3_s = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4_s = nn.Conv2d(128, 256, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers
        self.skipConv1 = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, 0),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
        )

        self.skipConv2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv3 = nn.Conv2d(64, 128, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer("res1_s", None)
        self.register_buffer("res2_s", None)
        self.register_buffer("res3_s", None)
        self.register_buffer("res4_s", None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        self.res1_s = self.relu(self.conv1_s(s))
        self.res2_s = self.relu(self.conv2_s(self.res1_s))
        self.res3_s = self.relu(self.conv3_s(self.res2_s))
        self.res4_s = self.relu(self.conv4_s(self.res3_s))

        self.res1_s = self.res1_s.squeeze()
        self.res2_s = self.res2_s.squeeze()
        self.res3_s = self.res3_s.squeeze()
        self.res4_s = self.res4_s.squeeze()

    # x is the input uncompensated image, s is a 1x3x256x256 surface image
    def forward(self, x, *argv):
        s = torch.cat([argv[0], argv[1]], 1)

        # surface feature extraction
        res1_s = self.relu(self.conv1_s(s)) if self.res1_s is None else self.res1_s
        res2_s = self.relu(self.conv2_s(res1_s)) if self.res2_s is None else self.res2_s
        res3_s = self.relu(self.conv3_s(res2_s)) if self.res3_s is None else self.res3_s
        res4_s = self.relu(self.conv4_s(res3_s)) if self.res4_s is None else self.res4_s

        # backbone
        # res1 = self.skipConv1(x)
        res1 = self.skipConv1(argv[0])
        x = self.relu(self.conv1(x) + res1_s)
        res2 = self.skipConv2(x)
        x = self.relu(self.conv2(x) + res2_s)
        res3 = self.skipConv3(x)
        x = self.relu(self.conv3(x) + res3_s)
        x = self.relu(self.conv4(x) + res4_s)
        x = self.relu(self.conv5(x) + res3)
        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = self.relu(self.conv6(x) + res1)

        x = torch.clamp(x, max=1)

        return x


class APRIF_PCNet_clamp(nn.Module):
    # Project-and-Capture Network for SPAA
    def __init__(
        self,
        mask=torch.ones((1, 3, 256, 256)),
        clamp_min=0,
        clamp_max=1,
        shading_net=None,
        fix_shading_net=False,
        use_mask=True,
        use_rough=True,
    ):
        super(APRIF_PCNet_clamp, self).__init__()
        self.name = self.__class__.__name__
        self.use_mask = use_mask
        self.use_rough = use_rough
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        if not self.use_mask:
            self.name += "_no_mask"
        if not self.use_rough:
            self.name += "_no_rough"

        # initialize from existing models or create new models
        # self.warping_net = copy.deepcopy(warping_net.module) if warping_net is not None else WarpingNet()
        self.shading_net = (
            copy.deepcopy(shading_net.module)
            if shading_net is not None
            else ShadingNetSPAA_clamp()
        )

        if self.use_mask:
            self.register_buffer("mask", mask.clone())

        # fix procam net
        for param in self.shading_net.parameters():
            param.requires_grad = not fix_shading_net

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        # self.warping_net.simplify(s)
        # self.shading_net.simplify(self.warping_net(s))
        self.shading_net.simplify(s)

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet
        # x = self.warping_net(x)

        if self.use_mask:
            x = x * self.mask
        if self.use_rough:
            x = self.shading_net(x, s, x * s, self.clamp_min, self.clamp_max)
        else:
            x = self.shading_net(x, s)

        return x


def predict_image(surface_tensor, projection_tensor, predict_model, device):
    predict_model.eval()

    with torch.no_grad():
        output = predict_model(projection_tensor, surface_tensor)

    return output


def morph():
    # num_threads = 20  # Adjust according to the number of cores on your machine
    # tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    # tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    # tf.config.optimizer.set_jit(True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        help="Source file name",
        default=join("./data/NST", surface, "uncmp/img_0001.png"),
        type=str,
    )
    parser.add_argument(
        "-t",
        "--target",
        help="Target file name",
        default=join("./data/NST", surface, "uncmp", "img_{:04d}.png".format(frame)),
        type=str,
    )
    args = parser.parse_args()

    if not args.source:
        print("No source file provided!")
        exit()

    if not args.target:
        print("No target file provided!")
        exit()

    dom_a = cv2.imread(args.source, cv2.IMREAD_COLOR)
    dom_b = cv2.imread(args.target, cv2.IMREAD_COLOR)

    # Checks if input and destination image are of the same dimensions.
    if dom_a.shape[1] != dom_b.shape[1] or dom_a.shape[0] != dom_b.shape[0]:
        print("Input Image is not the same dimensions as Destination Image.")
        sys.exit()

    # Store original height and width
    ORIG_WIDTH = dom_a.shape[1]
    ORIG_HEIGHT = dom_a.shape[0]

    dom_a = cv2.cvtColor(dom_a, cv2.COLOR_BGR2RGB)
    dom_a = cv2.resize(dom_a, (im_sz, im_sz), interpolation=cv2.INTER_AREA)
    dom_a = dom_a / 127.5 - 1

    dom_b = cv2.cvtColor(dom_b, cv2.COLOR_BGR2RGB)
    dom_b = cv2.resize(dom_b, (im_sz, im_sz), interpolation=cv2.INTER_AREA)
    dom_b = dom_b / 127.5 - 1

    origins = dom_a.reshape(1, im_sz, im_sz, 3).astype(np.float32)
    targets = dom_b.reshape(1, im_sz, im_sz, 3).astype(np.float32)

    preds_shared_memory = produce_warp_maps(origins, targets)
    use_warp_maps(origins, targets, frame, preds_shared_memory)

    if preds_shared_memory:
        del preds_shared_memory


def get_compensation_images(data_root, cam_surf, surface, model, fps, mode):
    torch.cuda.empty_cache()

    # Load data (from disk)
    # prj_valid_path = join(data_root, 'NST', surface, 'uncmp')
    # prj_valid = readImgsMT(prj_valid_path).to(device)
    # Load data (from memory)
    NEW_HEIGHT = 720
    NEW_WIDTH = 1280

    # Convert numpy array to PyTorch tensor
    tensor_shared_array = torch.from_numpy(np_shared_array).float()

    # Enlarge the image
    tensor_resized_array = torch.nn.functional.interpolate(
        tensor_shared_array.permute(0, 3, 1, 2),
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )

    # Initialize black background image
    background = torch.zeros((frame, 3, NEW_HEIGHT, NEW_WIDTH), dtype=torch.float32)

    # Center the enlarged image in the background image
    start_y = (NEW_HEIGHT - 512) // 2
    start_x = (NEW_WIDTH - 512) // 2
    background[:, :, start_y : start_y + 512, start_x : start_x + 512] = (
        tensor_resized_array
    )

    desire_test = background.div(255).to(device)

    compen_nest_pp = model

    if mode == "warp":
        cam_surf_texture = (
            torch.tensor(cam_surf, dtype=torch.float32).unsqueeze(0).to(device)
        )
        cam_surf_texture = cam_surf_texture.expand_as(desire_test).to(device)
        with torch.no_grad():
            # simplify CompenNet++
            compen_nest_pp.module.simplify(cam_surf_texture[0, ...].unsqueeze(0))

            # compensate using CompenNet++
            compen_nest_pp.eval()
            _, warp_cam_surf, _ = compen_nest_pp(desire_test, cam_surf_texture)
        del desire_test, cam_surf_texture

        return warp_cam_surf
    if mode == "compensation":
        cam_surf_test = cam_surf.expand_as(desire_test).to(device)
        with torch.no_grad():
            # simplify CompenNet++
            compen_nest_pp.module.simplify(cam_surf_test[0, ...].unsqueeze(0))

            # compensate using CompenNet++
            compen_nest_pp.eval()
            prj_cmp_test, _, _ = zip(
                *[
                    compen_nest_pp(desire_test[i : i + 1], cam_surf_test[i : i + 1])
                    for i in tqdm(
                        range(desire_test.size(0)), desc="Morph Compensating", unit="it"
                    )
                ]
            )  # compensated prj input image x^{*}
            prj_cmp_test = torch.cat(prj_cmp_test, dim=0)
        del desire_test, cam_surf_test

        # create image save path
        prj_cmp_path = join(data_root, "NST", surface, "cmp")
        if not os.path.exists(prj_cmp_path):
            os.makedirs(prj_cmp_path)

        # save images
        saveImgs(
            prj_cmp_test, prj_cmp_path, fps
        )  # compensated testing images, i.e., to be projected to the surface


class ContentLoss(nn.Module):
    def __init__(self, target_img, layers=["21"]):
        super(ContentLoss, self).__init__()
        self.target = target_img.detach()  # target image as a constant tensor
        self.vgg = torchvision.models.vgg19(pretrained=True).features.eval()
        self.layers = layers

        # Extract features for the target image
        self.target_features = self.extract_features(self.target)

    def extract_features(self, img):
        features = []
        x = img
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features

    def forward(self, input_img):
        input_features = self.extract_features(input_img)
        loss = sum(
            nn.functional.mse_loss(input_f, target_f)
            for input_f, target_f in zip(input_features, self.target_features)
        )
        return loss

    def update_target(self, target_img):
        self.target = target_img.detach()
        self.target_features = self.extract_features(self.target)


def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    resetRNGseed(0)
    cmp_path = join("./data/CMP/", f"{surface_name}_compen.pth")

    # Loading the style transfer model
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    optimizer = optim.Adam(pipe.unet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Load projector compensation model
    warping_net = Models.WarpingNet(
        with_refine="w/o_refine" not in "CompenNeSt++_pre-trained"
    )

    # with open(join('./data/CMP', f'{surface}_affine_mat.pkl'), 'rb') as f:
    #     affine_mat = pickle.load(f)
    #
    # warping_net.set_affine(affine_mat.flatten())
    # warping_net.eval()

    if torch.cuda.device_count() >= 1:
        warping_net = nn.DataParallel(warping_net).to(device)

    compen_nest = Models.CompenNeSt()
    state_dict = torch.load(
        "./_Compensation/checkpoint/blender_pretrained_CompenNeSt_l1+ssim_50000_32_20000_0.0015_0.8_2000_0.0001_20000.pth"
    )

    # Remove the `module.` prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    compen_nest.load_state_dict(new_state_dict)
    if torch.cuda.device_count() >= 1:
        compen_nest = nn.DataParallel(compen_nest).to(device)

    compen_nest_pp = Models.CompenNeStPlusplus(warping_net, compen_nest)
    model_checkpoint = cmp_path
    state_dict = torch.load(model_checkpoint, map_location=device)

    # Remove the `module.` prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    compen_nest_pp.load_state_dict(new_state_dict)

    if torch.cuda.device_count() >= 1:
        compen_nest_pp = nn.DataParallel(compen_nest_pp).to(device)

    compen_nest_pp.to(device)

    # Preload the projection surface
    cam_ref_path = join(data_root, "CMP", surface, "cam/raw/ref")
    cam_ref_crop_path = join(data_root, "CMP", surface, "cam/raw/ref_crop")

    cam_surf_d = readImgsMT(cam_ref_crop_path, index=[0]).to(device)
    clamp_min = cam_surf_d
    cam_surf_l = readImgsMT(cam_ref_crop_path, index=[1]).to(device)
    clamp_max = cam_surf_l
    cam_surf = readImgsMT(cam_ref_path, index=[2]).to(device)

    # Relit model
    predict_model = APRIF_PCNet_clamp(clamp_min=clamp_min, clamp_max=clamp_max)
    if torch.cuda.device_count() >= 1:
        predict_model = nn.DataParallel(predict_model, device_ids=[0]).to(device)
    predict_model.load_state_dict(torch.load(f"./data/CMP/{surface_name}_relit.pth"))
    predict_model.to(device)

    # surface light image
    img = join("./data/CMP", surface, "cam/raw/ref_crop", "img_0002.png")
    dest_img = join(
        "./data/NST",
        surface,
        "uncmp",
        "img_{:04d}.png".format(config.get("morph").get("frame")),
    )
    shutil.copyfile(img, dest_img)

    surface_gery_image = readImgsMT(cam_ref_crop_path, index=[2]).to(device)

    cal_content_loss = ContentLoss(cam_surf_l.cpu())
    # cal_style_loss = StyleLoss()

    while True:
        tick = 0

        loss_values = np.zeros(101)
        sat_loss_values = np.zeros(101)
        loss_1_values = np.zeros(101)
        loss_2_values = np.zeros(101)

        # Copy the last photo of the previous round to the first photo of the new round
        img = join(
            "./data/NST",
            surface,
            "uncmp",
            "img_{:04d}.png".format(config.get("morph").get("frame")),
        )
        dest_img = join("./data/NST", surface, "uncmp/img_0001.png")
        shutil.copyfile(img, dest_img)

        prompt = input("Transfer Prompt：")
        while True:
            # Utilize LGST
            image = PIL.Image.open(img).convert("RGB").resize((256, 256))
            images = pipe(
                prompt,
                image=image,
                num_inference_steps=config.get("transfer").get("infer_steps"),
                image_guidance_scale=1,
            ).images
            # images = pipe(prompt, image=image).images
            output_dir = join(
                "./data/NST",
                surface,
                "uncmp",
                "img_{:04d}.png".format(config.get("morph").get("frame")),
            )
            images[0].save(output_dir)

            desired_projection_path = join(
                "./data/NST",
                surface,
                "uncmp",
                "img_{:04d}.png".format(config.get("morph").get("frame")),
            )

            desired_projection = cv2.imread(desired_projection_path, cv2.IMREAD_COLOR)
            desired_projection = cv2.cvtColor(desired_projection, cv2.COLOR_BGR2RGB)
            desired_projection = cv2.resize(desired_projection, (256, 256))

            desired_projection2 = (
                torch.from_numpy(desired_projection)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255)
            )

            desired_projection3 = torch.nn.functional.interpolate(
                desired_projection2,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )

            # Initialize black background image
            background = torch.zeros((1, 3, 720, 1280), dtype=torch.float32)

            # Center the enlarged image in the background image
            start_y = (720 - 512) // 2
            start_x = (1280 - 512) // 2
            background[:, :, start_y : start_y + 512, start_x : start_x + 512] = (
                desired_projection3
            )

            desired_projection3 = background.to(device)

            cam_surf_texture = cam_surf.expand_as(desired_projection3).to(device)

            with torch.no_grad():
                compen_nest_pp.module.simplify(cam_surf_texture[0, ...].unsqueeze(0))
                # compensate using CompenNet++
                compen_nest_pp.eval()

                compensated_input, _, satloss = compen_nest_pp(
                    desired_projection3, cam_surf_texture
                )  # sat_loss

            compensated_input = torch.tensor(compensated_input[0]).to(device)

            # Convert tensors from PyTorch to NumPy arrays for use with OpenCV
            compensated_input_np = (
                compensated_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
            )

            # Use OpenCV to perform perspective transformation and convert the result directly to a PyTorch tensor
            compensated_input = (
                torch.from_numpy(
                    cv2.warpPerspective(
                        (compensated_input_np * 255).astype(np.uint8), M, (256, 256)
                    )
                )
                .float()
                .to(device)
                .div(255)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            predicted_image = predict_image(
                surface_gery_image, compensated_input, predict_model, device
            )

            desired_projection2 = desired_projection2.to(device)

            # content_loss = cal_content_loss(desired_projection2.cpu())
            # content_loss_values[tick] = content_loss
            # print(f'content_loss: {content_loss}')

            # style_loss = cal_style_loss(prompt, desired_projection2.cpu())
            # style_loss_values[tick] = style_loss
            # print(f'style_loss: {style_loss}')

            loss_1 = torch.mean((predicted_image - desired_projection2) ** 2)
            loss_1_values[tick] = loss_1
            print(f"loss_1: {loss_1}")

            # # loss_2
            # Use clamp to limit the range of desired_projection2
            clamped_projection = torch.clamp(
                desired_projection2, min=clamp_min, max=clamp_max
            )

            # lower_diff and upper_diff are combined into the same loss
            loss_2 = torch.mean((clamped_projection - desired_projection2) ** 2)

            loss_2_values[tick] = loss_2
            print(f"loss_2: {loss_2}")

            sat_loss = torch.tensor(satloss)
            sat_loss_values[tick] = satloss
            print(f"Saturation Loss: {sat_loss}")

            # total_loss = (loss_1).clone().detach().requires_grad_(True)
            # total_loss = (loss_2).clone().detach().requires_grad_(True)
            # total_loss = (sat_loss).clone().detach().requires_grad_(True)
            # total_loss = ((loss_1 + loss_2)).clone().detach().requires_grad_(True)
            # total_loss = ((loss_1 + sat_loss)).clone().detach().requires_grad_(True)
            # total_loss = ((loss_2 + sat_loss)).clone().detach().requires_grad_(True)
            total_loss = (
                ((loss_1 + loss_2 + sat_loss)).clone().detach().requires_grad_(True)
            )
            print(f"Loss: {total_loss}")

            loss_values[tick] = total_loss

            predicted_image = predicted_image.squeeze().cpu().numpy()
            predicted_image = np.transpose(predicted_image, (1, 2, 0))

            # Restore the image to its original pixel value range (0 to 255)
            predicted_image = predicted_image * 255.0
            predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                join("./data/NST", surface, "uncmp", "infer.png"),
                predicted_image.astype(np.uint8),
            )

            if tick == 0:
                first_loss = total_loss

            if total_loss <= first_loss:
                print("Iter: ", tick)
                cv2.imwrite(
                    join("./data/NST", surface, "uncmp", "infer.png"),
                    predicted_image.astype(np.uint8),
                )
                time.sleep(0.1)
                break

            if tick >= 100:
                # average_sat_loss = np.mean(sat_loss_values)
                # print("Average Sat loss:", average_sat_loss)
                # average_loss_1 = np.mean(loss_1_values)
                # print("Average Loss 1:", average_loss_1)
                # # average_loss_2 = np.mean(loss_2_values)
                # # print("Average Loss 2:", average_loss_2)
                # average_loss = np.mean(loss_values)
                # print("Average loss:", average_loss)
                cv2.imwrite(
                    join("./data/NST", surface, "uncmp", "infer.png"),
                    predicted_image.astype(np.uint8),
                )
                break

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Step the scheduler
            scheduler.step(total_loss)

            img = join("./data/CMP", surface, "cam/raw/ref_crop", "img_0002.png")
            dest_img = join(
                "./data/NST",
                surface,
                "uncmp",
                "img_{:04d}.png".format(config.get("morph").get("frame")),
            )
            shutil.copyfile(img, dest_img)

            tick += 1

        plt.figure(figsize=(10, 6))
        plt.plot(sat_loss_values, label="Sat Loss")
        plt.plot(loss_1_values, label="Loss 1")
        plt.plot(loss_2_values, label="Loss 2")
        # plt.plot(content_loss_values, label='Content Loss')
        # plt.plot(style_loss_values, label='Style Loss')
        plt.plot(loss_values, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Losses Over Epochs")
        plt.legend()
        plt.show()

        # image morphing
        morph()

        # utilize projector compensation
        get_compensation_images(
            data_root, cam_surf, surface, compen_nest_pp, fps, "compensation"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = "config.yaml"
    config = load_yaml(config_path)

    ORIG_WIDTH = config.get("morph").get("orig_width")
    ORIG_HEIGHT = config.get("morph").get("orig_height")

    TRAIN_EPOCHS = config.get("morph").get("train_epochs")
    add_scale = config.get("morph").get("add_scale")
    mult_scale = config.get("morph").get("mult_scale")
    warp_scale = config.get("morph").get("warp_scale")
    add_first = config.get("morph").get("add_first")

    frame = config.get("morph").get("frame")
    fps = config.get("compensation").get("fps")

    surface = config.get("procams").get("surface")
    surface_name = config.get("procams").get("surface")

    data_root = "./data"
    model_checkpoint = "./data/CMP/compensation.pth"

    # Creating a shared memory array
    shape = (frame, ORIG_HEIGHT, ORIG_WIDTH, 3)
    size = int(np.prod(shape))
    shared_array = Array("B", size)
    np_shared_array = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape(
        shape
    )

    # Convert shared memory array to PyTorch tensor and transfer to GPU
    # Note: This assumes that the dtype of np_shared_array is np.uint8
    # If your data type is different, please ensure the consistency of data type
    np_shared_array_float = (
        np_shared_array.astype(np.float32) / 255.0
    )  # Convert to floating point and normalize
    tensor_shared_array = torch.from_numpy(np_shared_array_float).cuda()

    main()
