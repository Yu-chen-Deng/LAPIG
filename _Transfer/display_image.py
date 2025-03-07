import os
from os.path import join
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None


def toggle_fullscreen(event, fig):
    if event.key == "escape":
        fig.canvas.manager.full_screen_toggle()


def update_plot_window(file_path, fps, surface):
    # Create a figure without a toolbar and border
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.toolbar.setVisible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("black")  # Set the background color to black
    ax.set_facecolor("black")  # Set the background color of the axis to black
    ax.axis("off")

    # Binding key events
    fig.canvas.mpl_connect(
        "key_press_event", lambda event: toggle_fullscreen(event, fig)
    )

    img = Image.open(join("./data/CMP", surface, "prj/cmp", "img_0001.png"))
    imshow_obj = ax.imshow(img)

    plt.show(block=False)

    prev_image = None

    while True:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    image_path = f.read().strip()
                    if image_path:
                        try:
                            with open(image_path, "rb") as image_file:
                                img_data = image_file.read()
                            img = Image.open(BytesIO(img_data))
                        except OSError as e:
                            if "Truncated File Read" in str(e):
                                continue
                            else:
                                raise e

                        if img != prev_image:
                            imshow_obj.set_data(img)
                            fig.canvas.draw_idle()
                            prev_image = img
            except Exception as e:
                pass
        plt.pause(1 / fps)


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_yaml(config_path)

    fps = config.get("compensation").get("fps")
    surface = config.get("procams").get("surface")

    image_path_file = "image_path.txt"  # File storing the latest image path
    update_plot_window(image_path_file, fps, surface)
