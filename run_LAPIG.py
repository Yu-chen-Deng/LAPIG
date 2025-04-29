import os
from os.path import join

from utils_LAPIG import load_config, get_config, check_and_start_visdom, execute

if __name__ == "__main__":
    execute("utils/register_utils_LAPIG.py")

    config_path = "config.yaml"
    config = load_config(config_path)

    setup = get_config(config, "setup_name")

    frame = get_config(config, "morph", "frame")

    if get_config(config, "AUTO_RUN_VISDOM") is True:
        check_and_start_visdom(port=8097)

    compen_ckpt_path = join(
        "./data/Compen+Relit/", setup, f"{setup}_CompenNeSt_pre-trained.pth"
    )

    relit_ckpt_path = join(
        "./data/Compen+Relit/", setup, f"{setup}_PSA_relit_PCNet.pth"
    )
    if not os.path.exists(compen_ckpt_path):
        # Project-and-capture
        execute(
            "./_ProCams/capture.py",
            args=["./data/Compen+Relit", "['ref', 'test', 'train']"],
        )

        # Compensation
        execute("./_Compen+Relit/src/python/train_pre-trained_compenNeSt.py")

        # Project-and-capture
        execute(
            "./_ProCams/capture.py",
            args=[
                "./data/Compen+Relit",
                "['cmp']",
            ],
        )

        pass

    if not os.path.exists(relit_ckpt_path):
        # Compensation Relighting
        execute("./_Compen+Relit/src/python/train_PSA_relit.py")
        pass

    if os.path.exists(compen_ckpt_path) and os.path.exists(relit_ckpt_path):
        # Editor Projection
        execute("./_LGST/editor_instructpix2pix.py", wait=True)
