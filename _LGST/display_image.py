import matplotlib.pyplot as plt
import time


def toggle_fullscreen(event, fig):
    if event.key == "escape":
        fig.canvas.manager.full_screen_toggle()


# def update_plot_window(fig, imshow_obj, tensor_data, fps):
#     # Convert tensor to numpy (assuming the tensor is in [0, 1] range)
#     tensor_data = (
#         tensor_data.permute(0, 2, 3, 1).cpu().numpy()
#     )  # Convert to [batch, h, w, c]

#     batch_idx = 0
#     while batch_idx < tensor_data.shape[0]:
#         # Update the image in the plot
#         imshow_obj.set_data(tensor_data[batch_idx])
#         fig.canvas.draw_idle()

#         # Update the batch index and pause to simulate fps
#         batch_idx += 1
#         plt.pause(1 / fps)  # control the frame rate


def update_plot_window(fig, imshow_obj, tensor_data, fps):
    tensor_data = tensor_data.permute(0, 2, 3, 1).cpu().numpy()

    ax = imshow_obj.axes
    fig.canvas.draw()

    background = fig.canvas.copy_from_bbox(ax.bbox)

    batch_idx = 0
    desired_frame_time = 1.0 / fps

    while batch_idx < tensor_data.shape[0]:
        frame_start = time.time()

        imshow_obj.set_data(tensor_data[batch_idx])

        fig.canvas.restore_region(background)
        ax.draw_artist(imshow_obj)
        fig.canvas.blit(ax.bbox)

        fig.canvas.flush_events()

        frame_duration = time.time() - frame_start
        sleep_time = max(0.0, desired_frame_time - frame_duration)
        time.sleep(sleep_time)

        batch_idx += 1


def create_plot_window(surface):
    surface = surface.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

    # Create a figure without a toolbar and border
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    fig.canvas.manager.toolbar.setVisible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor("black")  # Set the background color of the axis to black
    ax.axis("off")

    # Binding key events
    fig.canvas.mpl_connect(
        "key_press_event", lambda event: toggle_fullscreen(event, fig)
    )

    # Initialize display object
    imshow_obj = ax.imshow(surface, animated=True)  # Display the first image initially
    plt.show(block=False)

    return fig, imshow_obj
