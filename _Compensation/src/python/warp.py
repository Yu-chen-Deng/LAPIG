import cv2
import numpy as np
import os
import yaml


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

config_path = "config.yaml"
config = load_yaml(config_path)
surface_name = config.get("procams").get("surface")

# 用于存储四个选定的点,注意（全顺时针或逆时针）
points = []


def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select 4 Points", image)


def warp_image(image, corners, target_size=(512, 512)):
    target_corners = np.array(
        [
            [0, 0],
            [target_size[0] - 1, 0],
            [target_size[0] - 1, target_size[1] - 1],
            [0, target_size[1] - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(corners, target_corners)
    print("Perspective Transform Matrix (M):\n", M)  # 打印透视变换矩阵
    warped = cv2.warpPerspective(image, M, target_size)
    return warped


def process_images_in_directory(directory_path, corners, target_size=(512, 512)):
    for filename in os.listdir(directory_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            final_image = warp_image(image, corners, target_size)
            output_path = os.path.join(directory_path, "../cmpwowarp", f"{filename}")
            cv2.imwrite(output_path, final_image)
            print(f"Processed {filename} and saved to {output_path}")


# 设置图像目录路径和目标尺寸
directory_path = f"./data/CMP/{surface_name}/prj/cmp/"
target_size = (256, 256)

# 读取目录中的第一张图像用于手动选择点
first_image_path = os.path.join(directory_path, os.listdir(directory_path)[0])
image = cv2.imread(first_image_path)
clone = image.copy()

cv2.imshow("Select 4 Points", image)
cv2.setMouseCallback("Select 4 Points", select_points)

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) == 4:
    # 将选定的点转换为float32格式
    corners = np.array(points, dtype="float32")

    # 对目录中的所有图像进行处理
    process_images_in_directory(directory_path, corners, target_size)
else:
    print("Error: Please select exactly 4 points.")
