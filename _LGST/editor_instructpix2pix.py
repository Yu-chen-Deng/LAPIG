import os
from os.path import join

from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import cv2
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
import numpy as np
import matplotlib.pyplot as plt

from DiffMorph import *

from utils_LAPIG import (
    load_config,
    get_config,
    resetRNGseed,
    PIL2Tensor4D,
    Tensor4D2PIL,
    readImgsMT,
    saveImgs,
)
from CompenModels import CompenNeSt
from RelitModels import PSA_relit_PCNet

from display_image import *

plot_on = False


def sim_cap_compen(surface_tensor, projection_tensor, PSA_relit):
    PSA_relit.eval()

    with torch.no_grad():
        output = PSA_relit(projection_tensor, surface_tensor)

    return output


def main():
    resetRNGseed(0)

    # Preload the projection surface
    cam_ref_path = join(data_root, "Compen+Relit", surface, "cam/warp/ref")

    I_minor = readImgsMT(cam_ref_path, index=[0]).to(device)
    I_plus = readImgsMT(cam_ref_path, index=[1]).to(device)
    cam_surf = readImgsMT(cam_ref_path, index=[2]).to(device)

    fig, imshow_obj = create_plot_window(torch.ones(1, 3, 256, 256))

    # Preload LGST model
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    optimizer = optim.Adam(pipe.unet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Load Compensation model
    compen_nest = CompenNeSt()
    if torch.cuda.device_count() >= 1:
        compen_nest = nn.DataParallel(compen_nest, device_ids=[0]).to(device)
    compen_nest.load_state_dict(
        torch.load(
            f"./data/Compen+Relit/{surface}/{surface}_CompenNeSt_pre-trained.pth"
        )
    )
    compen_nest.to(device)

    # simplify + eval
    compen_nest.eval()
    compen_nest.module.simplify(cam_surf[0, ...].unsqueeze(0))

    # Preload PSA Relit model
    PSA_relit = PSA_relit_PCNet(clamp_min=I_minor, clamp_max=I_plus)
    if torch.cuda.device_count() >= 1:
        PSA_relit = nn.DataParallel(PSA_relit, device_ids=[0]).to(device)
    PSA_relit.load_state_dict(
        torch.load(f"./data/Compen+Relit/{surface}/{surface}_PSA_relit_PCNet.pth")
    )
    PSA_relit.to(device)

    frame_start = I_plus
    while True:
        tick = 0

        if plot_on:
            loss_values = np.zeros(101)
            loss_ps_values = np.zeros(101)
            loss_pc_values = np.zeros(101)
            loss_cs_values = np.zeros(101)

        prompt = input("User input text prompt：")
        while True:
            # Utilize LGST
            images = pipe(
                prompt,
                image=Tensor4D2PIL(I_plus),
                num_inference_steps=config.get("transfer").get("infer_steps"),
                image_guidance_scale=1,
            ).images

            output_dir = join(
                "./data/LGST",
                surface,
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            images[0].save(join(output_dir, "styl_surf.png"))

            styl_surf = PIL2Tensor4D(images[0], device=device)

            with torch.no_grad():
                compen_nest.module.simplify(cam_surf[0, ...].unsqueeze(0))
                # compensate using CompenNet++
                compen_nest.eval()

                compen_input, loss_cs = compen_nest(styl_surf, cam_surf)  # sat_loss

            cap_compen = sim_cap_compen(cam_surf, compen_input, PSA_relit)

            loss_pc = torch.mean((cap_compen - styl_surf) ** 2)

            clamped_styl_surf = torch.clamp(styl_surf, min=I_minor, max=I_plus)
            loss_ps = torch.mean((clamped_styl_surf - styl_surf) ** 2)

            loss_cs = torch.tensor(loss_cs)

            total_loss = (
                ((loss_pc + loss_ps + loss_cs)).clone().detach().requires_grad_(True)
            )

            if plot_on:
                loss_pc_values[tick] = loss_pc
                print(f"Projection Consistency: {loss_pc}")

                loss_ps_values[tick] = loss_ps
                print(f"Projection saturation: {loss_ps}")

                loss_cs_values[tick] = loss_cs
                print(f"Compensation saturation: {loss_cs}")

                loss_values[tick] = total_loss
                print(f"Loss: {total_loss}")

            if tick == 0:
                first_loss = total_loss

            if total_loss <= first_loss / 2.0:
                print("Iter: ", tick)
                saveImgs(cap_compen, f"./data/LGST/{surface}", name="cap_compen.png")
                time.sleep(0.1)
                break

            if tick >= 100:
                saveImgs(cap_compen, f"./data/LGST/{surface}", name="cap_compen.png")
                break

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Step the scheduler
            scheduler.step(total_loss)

            tick += 1

        if plot_on:
            plt.figure(figsize=(10, 6))
            plt.plot(loss_pc_values, label="Projection Consistency")
            plt.plot(loss_ps_values, label="Projection Saturation")
            plt.plot(loss_cs_values, label="Compensation Saturation")
            plt.plot(loss_values, label="Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Losses Over Epochs")
            plt.legend()
            plt.show()

        # image morphing
        frame_end = styl_surf
        styl_frames = morph(frame_start, frame_end, config)
        frame_start = styl_surf

        # utilize projector compensation
        with torch.no_grad():
            styl_frames.copy_(compen_nest(styl_frames, cam_surf)[0])

        update_plot_window(fig, imshow_obj, styl_frames, fps)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = "config.yaml"
    config = load_config(config_path)

    fps = get_config(config, "compensation", "fps")

    surface = get_config(config, "setup_name")
    resize = get_config(config, "procams", "resize")

    data_root = "./data"
    model_checkpoint = "./data/Compen+Relit/compensation.pth"

    main()
