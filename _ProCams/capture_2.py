# Author: Bingyao Huang (https://github.com/BingyaoHuang/SPAA/)
# Modified by: Yuchen Deng

"""
Set up ProCams and capture data
"""
import copy
import os
import sys
import warnings

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join
from omegaconf import DictConfig
import utils as ut
import matplotlib.pyplot as plt
from greycode import *
import time
import yaml


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None


def init_prj_window(prj_w, prj_h, val, offset, sl=False):
    """
    Initialize the projector window using plt
    :param prj_w:
    :param prj_h:
    :param val:
    :param offset: move the projector window by an offset in (x, y) format
    :return:
    """
    # initial image
    im = np.ones((prj_h, prj_w, 3), np.uint8) * val
    if not sl:
        disp_size = min(prj_h, prj_w)
        im = cv.resize(im, (disp_size, disp_size))
    # create figure and move to projector screen and set to full screen
    plt_backend = plt.get_backend()
    fig = plt.figure()

    # uncheck pycharm scientific "Show plots in tool window" when "AttributeError: 'FigureCanvasInterAgg' object has no attribute 'window'"
    if "Qt" in plt_backend:
        fig.canvas.window().statusBar().setVisible(False)  # (Qt only)

    ax = plt.imshow(im, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    mng = plt.get_current_fig_manager()
    if "Qt" in plt_backend:
        mng.window.setGeometry(
            *offset, prj_w, prj_h
        )  # change the offsets according to your setup (Qt only)
        mng.full_screen_toggle()  # to full screen (TkAgg may not work well, and resets the window position to the primary screen)
    elif "Tk" in plt_backend:  # Windows only, and the frame rate is not stable
        mng.window.geometry(f"{prj_w}x{prj_h}+{offset[0]}+{offset[1]}")
        mng.window.overrideredirect(1)  # Windows only
        mng.window.state("zoomed")  # Windows only
        plt.pause(0.02)
    fig.show()

    return ax


def init_cam(cam_raw_sz=None):
    if sys.platform == "win32":
        # Initialize camera
        cam = cv.VideoCapture(
            0, cv.CAP_DSHOW
        )  # windows only to get rid of the annoying warning
    else:
        cam = cv.VideoCapture(0)

    if cam_raw_sz is not None:
        cam.set(cv.CAP_PROP_FRAME_WIDTH, cam_raw_sz[0])
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, cam_raw_sz[1])
    cam.set(
        cv.CAP_PROP_BUFFERSIZE, 1
    )  # set the buffer size to 1 to avoid delayed frames
    cam.set(
        cv.CAP_PROP_FPS, 60
    )  # set the max frame rate (cannot be larger than cam's physical fps) to avoid delayed frames
    time.sleep(2)

    if not cam.isOpened():
        print("Cannot open camera")
    return cam


def project_capture_data(prj_input_path, cam_cap_path, setup_info, raw=True, sl=False):
    print(f"Projecting {prj_input_path} and \ncapturing to {cam_cap_path}")
    # all sz are in (w, h) format
    (
        prj_screen_sz,
        prj_offset,
        cam_raw_sz,
        cam_crop_sz,
        cam_im_sz,
        delay_frames,
        delay_time,
    ) = (
        setup_info["prj_screen_sz"],
        setup_info["prj_offset"],
        setup_info["cam_raw_sz"],
        setup_info["cam_crop_sz"],
        setup_info["cam_im_sz"],
        setup_info["delay_frames"],
        setup_info["delay_time"],
    )
    if not os.path.exists(cam_cap_path):
        os.makedirs(cam_cap_path)

    # load projector input images
    im_prj = np.uint8(ut.torch_imread_mt(prj_input_path).permute(2, 3, 1, 0) * 255)
    prj_im_aspect = im_prj.shape[1] / im_prj.shape[0]
    prj_screen_aspect = prj_screen_sz[0] / prj_screen_sz[1]

    # check aspect ratio
    if math.isclose(prj_im_aspect, prj_screen_aspect, abs_tol=1e-3):
        warnings.warn(
            f"The projector input image aspect ratio {prj_im_aspect} is different from the screen aspect ratio {prj_screen_aspect}, "
            f"image will be resized to center fill the screen"
        )

    # initialize camera and project
    plt.close("all")
    prj = init_prj_window(*prj_screen_sz, 0.5, (prjXOffset, 0), sl)
    cam = init_cam(cam_raw_sz)

    # clear camera buffer
    for j in range(0, 100):
        _, im_cam = cam.read()

    num_im = im_prj.shape[-1]
    # num_im = 20 # for debug

    # project-and-capture, then save images
    for i in tqdm(range(num_im)):
        prj.set_data(im_prj[..., i])
        plt.pause(
            delay_time
        )  # adjust this according to your hardware and program latency
        plt.draw()

        # fetch frames from the camera buffer and drop [num_frame_delay] frames due to delay
        for j in range(0, delay_frames):
            _, im_cam = cam.read()
        if raw:
            cv.imwrite(
                join(cam_cap_path, "img_{:04d}.png".format(i + 1)),
                cv.resize(im_cam, (camWidth, camHeight), interpolation=cv.INTER_AREA),
            )
        # apply center crop and resize to the camera frames, then save to files
        # cv.imwrite(join(cam_cap_path, 'img_{:04d}.png'.format(i + 1)), cv.resize(cc(im_cam, cam_crop_sz), cam_im_sz, interpolation=cv.INTER_AREA))
        # cv.imwrite(join(cam_cap_path, 'img_{:04d}.png'.format(i + 1)), cv.resize(im_cam, cam_im_sz, interpolation=cv.INTER_AREA))

    # release camera and close projector windows
    cam.release()
    plt.close("all")


def create_gray_pattern(w, h):
    # Python implementation of MATLAB's createGrayPattern
    nbits = np.ceil(np.log2([w, h])).astype(
        int
    )  # # of bits for vertical/horizontal patterns
    offset = (2**nbits - [w, h]) // 2  # offset the binary pattern to be symmetric

    # coordinates to binary code
    c, r = np.meshgrid(np.arange(w), np.arange(h))
    bin_pattern = [
        np.unpackbits(
            (c + offset[0])[..., None].view(np.uint8),
            axis=-1,
            bitorder="little",
            count=nbits[0],
        )[..., ::-1],
        np.unpackbits(
            (r + offset[1])[..., None].view(np.uint8),
            axis=-1,
            bitorder="little",
            count=nbits[1],
        )[..., ::-1],
    ]

    # binary pattern to gray pattern
    gray_pattern = copy.deepcopy(bin_pattern)
    for n in range(len(bin_pattern)):
        for i in range(1, bin_pattern[n].shape[-1]):
            gray_pattern[n][:, :, i] = np.bitwise_xor(
                bin_pattern[n][:, :, i - 1], bin_pattern[n][:, :, i]
            )

    # allPatterns also contains complementary patterns and all 0/1 patterns
    prj_patterns = np.zeros((h, w, 2 * sum(nbits) + 2), dtype=np.uint8)
    prj_patterns[:, :, 0] = 1  # All ones pattern

    # Vertical
    for i in range(gray_pattern[0].shape[-1]):
        prj_patterns[:, :, 2 * i + 2] = gray_pattern[0][:, :, i].astype(np.uint8)
        prj_patterns[:, :, 2 * i + 3] = np.logical_not(gray_pattern[0][:, :, i]).astype(
            np.uint8
        )

    # Horizontal
    for i in range(gray_pattern[1].shape[-1]):
        prj_patterns[:, :, 2 * i + 2 * nbits[0] + 2] = gray_pattern[1][:, :, i].astype(
            np.uint8
        )
        prj_patterns[:, :, 2 * i + 2 * nbits[0] + 3] = np.logical_not(
            gray_pattern[1][:, :, i]
        ).astype(np.uint8)

    prj_patterns *= 255

    # to RGB image
    # prj_patterns = np.transpose(np.tile(prj_patterns[..., None], (1, 1, 3)), (0, 1, 3, 2))  # to (h, w, c, n)
    prj_patterns = np.transpose(
        np.tile(prj_patterns[..., None], (1, 1, 3)), (2, 0, 1, 3)
    )  # to (n, h, w, c)

    return prj_patterns


def process_images(input_folder):
    # Obtain parent directory
    parent_folder = os.path.dirname(input_folder)

    output_folder = os.path.join(
        parent_folder, f"{os.path.basename(input_folder)}_crop"
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
        ):
            filepath = os.path.join(input_folder, filename)
            with Image.open(filepath) as img:
                width, height = img.size
                left = (width - 512) / 2
                top = (height - 512) / 2
                right = (width + 512) / 2
                bottom = (height + 512) / 2
                cropped_img = img.crop((left, top, right, bottom))
                resized_img = cropped_img.resize((256, 256))

                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)


def main():
    # %% (1) [local] Setup configs, e.g., projector and camera sizes, and sync delay params, etc.
    cam_save_sz = (camWidth, camHeight)
    prj_screen_sz = (prjWidth, prjHeight)
    setup_info = DictConfig(
        dict(
            prj_screen_sz=prj_screen_sz,  # projector screen resolution (i.e., set in the OS)
            prj_im_sz=(
                256,
                256,
            ),  # projector input image resolution, the image will be scaled to match the prj_screen_sz by plt
            prj_offset=(
                2560,
                0,
            ),  # an offset to move the projector plt figure to the correct screen (check your OS display setting)
            cam_raw_sz=(1280, 720),  # the size of camera's direct output frame
            cam_crop_sz=(
                960,
                720,
            ),  # a size used to center crop camera output frame, cam_crop_sz <= cam_raw_sz
            cam_im_sz=(
                320,
                240,
            ),  # a size used to resize center cropped camera output frame, cam_im_sz <= cam_crop_sz, and should keep aspect ratio
            classifier_crop_sz=(
                240,
                240,
            ),  # a size used to center crop resize cam image for square classifier input, classifier_crop_sz <= cam_im_sz
            prj_brightness=0.5,  # brightness (0~1 float) of the projector gray image for scene illumination.
            # adjust the two params below according to your ProCams latency, until the projected and captured numbers images are correctly matched
            delay_frames=delay_frames,  # how many frames to drop before we capture the correct one, increase it when ProCams are not in sync
            delay_time=delay_time,  # a delay time (s) between the project and capture operations for software sync, increase it when ProCams are not in sync
        )
    )

    # Check projector and camera FOV (the camera should see the full projector FOV);
    # focus (fixed and sharp, autofocus and image stabilization off); aspect ratio;
    # exposure (lowest ISO, larger F-number and proper shutter speed); white balance (fixed); flickering (anti on) etc.
    # Make sure when projector brightness=0.5, the cam-captured image is correctly exposed and correctly classified by the classifiers.

    # create projector window with different brightnesses, and check whether the exposures are correct
    prj_fig = list()
    for brightness in [0, setup_info["prj_brightness"], 1.0]:
        prj_fig.append(
            ut.init_prj_window(
                *setup_info["prj_screen_sz"], brightness, setup_info["prj_offset"]
            )
        )

    print(
        'Previewing the camera, make sure everything looks good and press "q" to exit...'
    )
    ut.preview_cam(
        setup_info["cam_raw_sz"],
        (min(setup_info["cam_raw_sz"]), min(setup_info["cam_raw_sz"])),
    )
    # plt.close('all')

    # prj_sl_path = join(data_root, 'sl')
    # if not os.path.exists(prj_sl_path):
    #     os.makedirs(prj_sl_path)
    #
    # cam_sl_path = join(data_root, surface, 'cam/sl')
    # if not os.path.exists(cam_sl_path):
    #     os.makedirs(cam_sl_path)

    # Generate Gray code pattern and save and capture projection SL
    # im_gray_sl = create_gray_pattern(*setup_info['prj_screen_sz'])
    # ut.save_imgs(im_gray_sl, prj_sl_path)
    # project_capture_data(prj_sl_path, cam_sl_path, setup_info, sl=True)

    for folder in folders:
        prj_input_path = join(data_root, surface, "prj", folder)
        if not os.path.exists(prj_input_path):
            os.makedirs(prj_input_path)

        cam_raw_path = join(data_root, surface, "cam", "raw", folder)
        if not os.path.exists(cam_raw_path):
            os.makedirs(cam_raw_path)

        # cam_warp_path = join(data_root, surface, 'cam', 'warp', folder)
        # if not os.path.exists(cam_warp_path):
        #     os.makedirs(cam_warp_path)

        # Project and capture, then save the images to the respective paths
        project_capture_data(prj_input_path, cam_raw_path, setup_info, sl=False)

        process_images(cam_raw_path)
        cam_raw_ref_path = join(data_root, surface, "cam", "raw", "ref")
        process_images(cam_raw_ref_path)


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_yaml(config_path)

    camWidth = config.get("procams").get("cam_width")
    camHeight = config.get("procams").get("cam_height")
    prjWidth = config.get("procams").get("prj_width")
    prjHeight = config.get("procams").get("prj_height")
    prjXOffset = config.get("procams").get("prj_xoffset")
    delay_frames = config.get("procams").get(
        "delay_frames"
    )  # how many frames to drop before we capture the correct one, increase it when ProCams are not in sync
    delay_time = config.get("procams").get(
        "delay_time"
    )  # a delay time (s) between the project and capture operations for software sync, increase it when ProCams are not in sync
    reSize = config.get("procams").get("resize")

    data_root = "./data/CMP"
    surface = config.get("procams").get("surface")
    folders = ["cmp"]

    main()
