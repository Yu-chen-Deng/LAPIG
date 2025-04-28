# Obtained from https://github.com/volotat/DiffMorph (Author: Alexey Borsky)
# Modified by: Yuchen Deng

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from os.path import join

import tensorflow as tf
import numpy as np
import cv2
import sys
import torch
from tqdm import tqdm
import multiprocessing as mp

from utils_LAPIG import load_config, get_config

config = load_config("config.yaml")

# Morph image size
im_sz = 256
# Morph mapping size
mp_sz = 96

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

            # cv2.imwrite(
            #     "./_Morph/train/a_to_b_%d.jpg" % epoch,
            #     cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR),
            # )

            res_origins = tf.clip_by_value(res_origins, -1, 1)[0]
            res_img = ((res_origins.numpy() + 1) * 127.5).astype(np.uint8)

            res_img = cv2.resize(
                res_img, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_AREA
            )

            # cv2.imwrite(
            #     "./_Morph/train/b_to_a_%d.jpg" % epoch,
            #     cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR),
            # )

            # Save the weights file to disk
            # np.save('./_Morph/preds.npy', preds.numpy())
            # Save the weights file to memory
            preds_np = preds.numpy().flatten()

    shared_array = mp.Array("f", preds_np)

    return shared_array


# generating frames using deformation mapping
def use_warp_maps(origins, targets, steps, warp_maps):
    STEPS = steps

    # Load the weights file from disk
    # preds = np.load('./_Morph/preds.npy')
    # Load the weights file from memory
    preds_np = np.frombuffer(warp_maps.get_obj(), dtype=np.float32)
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
    # cv2.imwrite("./_Morph/morph/maps.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

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
        res = res_numpy[0].transpose(2, 0, 1)

        # img = ((res_numpy[0] + 1) * 127.5).astype(np.uint8)
        img = torch.tensor((res + 1) * 0.5).clamp(0, 1).to(device)
        styl_frames[i, :, :, :] = img
        # output_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # output_img = cv2.resize(
        #     output_img, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_AREA
        # )

        # save the uncompensated image to disk
        # cv2.imwrite(join("./data/LGST", surface, 'uncmp', 'img_{:04d}.png'.format(i+1)), output_img)

        # save the uncompensated image to memory
        # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        # np_shared_array[i] = output_img

        # if (i + 1) % 10 == 0:
        #     res_img[im_sz * 0 : im_sz * 1, i // 10 * im_sz : (i // 10 + 1) * im_sz] = (
        #         img
        #     )
        #     res_img[im_sz * 1 : im_sz * 2, i // 10 * im_sz : (i // 10 + 1) * im_sz] = (
        #         (res_targets.numpy()[0] + 1) * 127.5
        #     ).astype(np.uint8)
        #     res_img[im_sz * 2 : im_sz * 3, i // 10 * im_sz : (i // 10 + 1) * im_sz] = (
        #         (res_origins.numpy()[0] + 1) * 127.5
        #     ).astype(np.uint8)

    # cv2.imwrite("./_Morph/morph/result.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
    # cv2.destroyAllWindows()


def morph(frame_start, frame_end, config):
    # num_threads = 20  # Adjust according to the number of cores on your machine
    # tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    # tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    # tf.config.optimizer.set_jit(True)
    global ORIG_WIDTH, ORIG_HEIGHT, TRAIN_EPOCHS, add_scale, mult_scale, warp_scale, add_first
    global im_sz, mp_sz
    global styl_frames

    ORIG_WIDTH = get_config(config, "morph", "orig_width")
    ORIG_HEIGHT = get_config(config, "morph", "orig_height")

    TRAIN_EPOCHS = get_config(config, "morph", "train_epochs")
    add_scale = get_config(config, "morph", "add_scale")
    mult_scale = get_config(config, "morph", "mult_scale")
    warp_scale = get_config(config, "morph", "warp_scale")
    add_first = get_config(config, "morph", "add_first")

    frames = get_config(config, "morph", "frame")
    resize = get_config(config, "procams", "resize")

    styl_frames = torch.empty(frames, 3, resize, resize).to(device)

    dom_a = frame_start.squeeze(0)
    dom_a = dom_a[[2, 1, 0], :, :]
    dom_a = (dom_a * 255).clamp(0, 255).byte().cpu().numpy()
    dom_a = dom_a.transpose(1, 2, 0)

    dom_b = frame_end.squeeze(0)
    dom_b = dom_b[[2, 1, 0], :, :]
    dom_b = (dom_b * 255).clamp(0, 255).byte().cpu().numpy()
    dom_b = dom_b.transpose(1, 2, 0)
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

    warp_maps = produce_warp_maps(origins, targets)
    use_warp_maps(origins, targets, frames, warp_maps)

    if warp_maps:
        del warp_maps

    return styl_frames
