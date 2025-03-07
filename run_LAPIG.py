import os
from os.path import join
import subprocess
import time
import sys
import yaml
import psutil


def is_visdom_running():
    for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmdline = process.info["cmdline"]
            if cmdline and "visdom.server" in " ".join(cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False


def start_visdom():
    if not is_visdom_running():
        print("Visdom server is not running. Starting now...")
        subprocess.Popen(
            ["python", "-m", "visdom.server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print("Visdom server is already running.")


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
        time.sleep(0.1)
        if process.poll() is not None:
            print(f"Failed to start {script_name}")
            sys.exit(0.1)
        return process
    except Exception as e:
        print(f"Error starting {script_name}: {e}")
        sys.exit(0.1)


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_yaml(config_path)

    frame = config.get("morph").get("frame")
    surface = config.get("procams").get("surface")

    start_visdom()

    cmp_path = join("./data/CMP/", f"{surface}_compen.pth")
    relit_path = join("./data/CMP/", f"{surface}_relit.pth")
    if not os.path.exists(cmp_path):
        # Project-and-capture
        prjcap_script = "./_ProCams/capture.py"
        prjcap_process = start_process(prjcap_script)
        prjcap_process.wait()

        # Compensation
        precmp_script = "./_Compensation/src/python/train_pre-trained_compenNeSt++.py"
        precmp_process = start_process(precmp_script)
        precmp_process.wait()

        # Warp
        warp_script = "./_Compensation/src/python/warp.py"
        warp_process = start_process(warp_script)
        warp_process.wait()

    if not os.path.exists(relit_path):
        # Project-and-capture
        # prjcap_script = "./_ProCams/capture_2.py"
        # prjcap_process = start_process(prjcap_script)
        # prjcap_process.wait()

        # Compensation Relighting
        precmp_script = "./_PSA_relit/src/python/train_PSA_relit.py"
        precmp_process = start_process(precmp_script)
        precmp_process.wait()

    if os.path.exists(cmp_path) and os.path.exists(relit_path):
        # Monitor Directory
        monitor_script = "./_Transfer/monitor_directory.py"
        monitor_process = start_process(monitor_script)

        # Display Window
        display_script = "./_Transfer/display_image.py"
        display_process = start_process(display_script)

        # Editor Projection
        editor_script = "./_Transfer/editor_instructpix2pix.py"
        editor_process = start_process(editor_script)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        monitor_process.terminate()
        display_process.terminate()
        editor_process.terminate()
        monitor_process.wait()
        display_process.wait()
        editor_process.wait()
