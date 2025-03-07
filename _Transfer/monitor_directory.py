import os
from os.path import join
import time
import glob
import threading
import yaml
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def clear_directory(directory):
    files = glob.glob(os.path.join(directory, '*'))
    for f in files:
        try:
            os.remove(f)
        except IsADirectoryError:
            # Handle directories if necessary
            pass
        except Exception as e:
            print(f'Failed to delete {f}. Reason: {e}')

class ImageHandler(FileSystemEventHandler):
    def __init__(self, file_path):
        self.file_path = file_path

    def on_created(self, event):
        if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            self.update_file(event.src_path)

    def on_modified(self, event):
        if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            self.update_file(event.src_path)

    def update_file(self, image_path):
        with open(self.file_path, 'w') as f:
            f.write(image_path)

def load_yaml(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

def monitor_directory(directory, file_path, fps):
    observer = Observer()
    handler = ImageHandler(file_path)
    observer.schedule(handler, directory, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1/fps)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_yaml(config_path)

    fps = config.get('compensation').get('fps')
    surface = config.get('procams').get('surface')

    directory_to_watch = join('./data/NST', surface, 'cmp')  # Replace with the directory path you want to monitor
    if not os.path.exists(directory_to_watch): os.makedirs(directory_to_watch)
    clear_directory(directory_to_watch)

    image_path_file = "image_path.txt"  # File storing the latest image path

    # Check if the file exists
    if os.path.exists(image_path_file):
        # Delete the file
        os.remove(image_path_file)

    file = open(image_path_file, 'w')
    file.close()

    monitor_thread = threading.Thread(target=monitor_directory, args=(directory_to_watch, image_path_file, fps))
    monitor_thread.daemon = True
    monitor_thread.start()

    refresh_time = (1 / fps) * 0.999
    while True:
        time.sleep(refresh_time)
