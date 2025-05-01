import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pygame
import math
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from pygame_widgets.button import Button
import datetime
import csv

# Initialize YOLO model
model = YOLO("Final/yolo11s.pt")
names = model.model.names

# OpenCV VideoCapture (Use a video file or webcam)
cap = cv2.VideoCapture("Final/vidp2.mp4")

count = 0
box_corner_annotator = sv.BoxAnnotator()

# Pygame initialization
pygame.init()
pygame.display.set_caption("Dashboard")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PINK = (255, 105, 180)
PURPLE = (255, 0, 255)

# 取得第一幀
ret, frame = cap.read()
frame = cv2.resize(frame, (1020, 600))

# 定義視窗的大小 (比圖像大)
window_width = 1500
window_height = 800

# 創建黑色背景
black_window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

# 計算圖像在黑色背景中的起始位置
start_y = (window_height - frame.shape[0]) // 2
start_x = (window_width - frame.shape[1]) // 2

print(frame.shape)
print(start_y, start_x)

# 將圖像放入黑色背景中
black_window[start_y : start_y + frame.shape[0], start_x : start_x + frame.shape[1]] = (
    frame
)

frame = black_window

if not ret:
    print("無法讀取影片或影片為空")
    exit()

# 用來保存標記點的列表
points = []


# 鼠標回調函數，用來捕捉點擊的坐標
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # 左鍵單擊事件
        # 在圖片上畫一個圓點，標示點的位置
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        points.append((x - 240, y - 100))  # 將點的位置存入列表
        print(f"點的位置: ({x}, {y})")

        # 顯示更新後的圖片
        cv2.imshow("Image", frame)


# 顯示圖片
cv2.imshow("Image", frame)

# 設置鼠標回調函數
cv2.setMouseCallback("Image", click_event)

# 等待使用者標記四個點
while len(points) < 4:
    cv2.waitKey(1)  # 持續等待事件

print("標記的點:", points)

# 關閉所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 將捕捉到的 points 轉換為 numpy 格式
src_points = np.float32(points)

# Pygame screen for dashboard
dashboard_width, dashboard_height = 400, 600
dashboard = pygame.display.set_mode((dashboard_width, dashboard_height))
font = pygame.font.Font(None, 30)  # Choose a font for text

# Bird's Eye View Parameters
bev_width, bev_height = 400, 600
# 定義第一人稱視圖中的四個點 (根據你的影片調整這些點)
# src_points2 = np.float32([[340, 330], [470, 330], [-900, 580], [1700, 580]])
# print(src_points2)
# 定義俯視圖中對應的四個點 (這些點構成一個矩形)
dst_points = np.float32(
    [[0, 0], [bev_width, 0], [0, bev_height], [bev_width, bev_height]]
)
# 計算單應性矩陣
H = cv2.getPerspectiveTransform(src_points, dst_points)
# 定義鳥瞰圖視窗的大小
bev_image_width, bev_image_height = 400, 600

# 預先加載音效檔
pygame.mixer.init()
warning_sound = pygame.mixer.Sound("Final/warning.mp3")
# 創建一個 Channel 對象
warning_channel = pygame.mixer.Channel(0)

# 距離閾值
WARNING_DISTANCE_THRESHOLD = 70

# 網格比例尺相關參數
GRID_SIZE_METERS = 1
PIXELS_PER_METER = 40

# CSV 文件名
csv_filename = "Final/object_data.csv"


# 寫入 CSV 文件的函數
def write_to_csv(data):
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # 如果文件是空的，寫入標題行
            writer.writerow(
                ["Timestamp", "Track ID", "Class", "Center X", "Center Y", "Distance"]
            )
        writer.writerow(data)


def draw_circle_with_border(
    surface, center, radius, border_color, fill_color, border_thickness=2
):
    pygame.draw.circle(surface, border_color, center, radius + border_thickness)
    pygame.draw.circle(surface, fill_color, center, radius)


# Function to display text on the dashboard
def display_text(surface, text, pos, color=WHITE):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=pos)
    surface.blit(text_surface, text_rect)


# OpenCV window setup
cv2.namedWindow("RGB")
cv2.namedWindow("Radar Image")
cv2.namedWindow("Bird's Eye View Image")
cv2.moveWindow("RGB", 0, 0)
cv2.moveWindow("Radar Image", 1020, 0)
cv2.moveWindow("Bird's Eye View Image", 1420, 0)
frame_width = 1020
frame_height = 600

# Variables for dashboard information
detected_objects_count = 0
last_warning_time = 0
show_grid = True
show_labels = True


# 滑塊設定
slider_width = 200
slider_height = 20
slider_x = dashboard_width - slider_width - 20
slider_y = 50

# 創建滑塊
warning_distance_slider = Slider(
    dashboard,
    slider_x,
    slider_y,
    slider_width,
    slider_height,
    min=10,
    max=150,
    step=1,
    initial=WARNING_DISTANCE_THRESHOLD,
    handleColour=WHITE,
    handleRadius=10,
    colour=GREEN,
)

# 創建顯示滑塊值的文本框
output_box = TextBox(
    dashboard,
    slider_x,
    slider_y + slider_height + 10,
    slider_width,
    30,
    fontSize=20,
    borderColour=WHITE,
    textColour=WHITE,
    onSubmit=lambda: None,  # 不做任何事情
    radius=5,
    borderThickness=1,
)
output_box.disable()  # 禁止用戶編輯

# 創建切換標籤顯示的按鈕
label_button = Button(
    dashboard,
    dashboard_width - 150,
    120,
    130,
    40,
    text="Toggle Labels",
    fontSize=25,
    margin=5,
    inactiveColour=(220, 220, 220),
    hoverColour=(200, 200, 200),
    pressedColour=(150, 150, 150),
    radius=5,
    onClick=lambda: globals().update({"show_labels": not show_labels}),
)

# 創建截圖按鈕
screenshot_button = Button(
    dashboard,
    dashboard_width - 150,
    180,
    130,
    40,
    text="Screenshot",
    fontSize=25,
    margin=5,
    inactiveColour=(220, 220, 220),
    hoverColour=(200, 200, 200),
    pressedColour=(150, 150, 150),
    radius=5,
    onClick=lambda: save_screenshot(),
)


def save_screenshot():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"Final/screenshot_main_{now}.png", annotated_frame)
    cv2.imwrite(f"Final/screenshot_radar_{now}.png", radar_image)
    cv2.imwrite(f"Final/screenshot_bev_{now}.png", bird_eye_view)
    print(f"Screenshots saved as screenshot_{now}.png")


# 創建切換網格顯示的按鈕
grid_button = Button(
    dashboard,
    dashboard_width - 150,
    240,  # 調整按鈕位置以避免重疊
    130,
    40,
    text="Toggle Grid",
    fontSize=25,
    margin=5,
    inactiveColour=(220, 220, 220),
    hoverColour=(200, 200, 200),
    pressedColour=(150, 150, 150),
    radius=5,
    onClick=lambda: globals().update({"show_grid": not show_grid}),
)

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model.track(frame[:500, :830], persist=True, classes=[0, 2, 3])

    bird_eye_view = np.zeros((bev_height + 50, bev_width, 3), dtype=np.uint8)
    radar_image = np.zeros((bev_image_height + 50, bev_image_width, 3), dtype=np.uint8)
    radar_image = cv2.warpPerspective(
        frame, H, (bev_image_width, bev_image_height + 50)
    )

    if show_grid:
        grid_size_pixels_vertical = int(GRID_SIZE_METERS * PIXELS_PER_METER)
        grid_size_pixels_horizontal = int(GRID_SIZE_METERS * PIXELS_PER_METER * 1.5)
        for y in range(0, bev_height, grid_size_pixels_vertical):
            cv2.line(bird_eye_view, (0, y), (bev_width, y), WHITE, 1)
            cv2.putText(
                bird_eye_view,
                f"{int((bev_height - y) / grid_size_pixels_vertical * GRID_SIZE_METERS)+1}m",
                (5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                WHITE,
                1,
            )
        for x in range(0, bev_width, grid_size_pixels_horizontal):
            cv2.line(bird_eye_view, (x, 0), (x, bev_height), WHITE, 1)
            cv2.putText(
                bird_eye_view,
                f"{math.ceil((x-bev_width/2) / grid_size_pixels_horizontal * GRID_SIZE_METERS)}m",
                (x + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                WHITE,
                1,
            )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        detections = sv.Detections(
            xyxy=boxes,
            class_id=np.array(class_ids),
            tracker_id=np.array(track_ids),
        )
        annotated_frame = box_corner_annotator.annotate(frame.copy(), detections)

        detected_objects_count = len(detections)

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            x1, y1, x2, y2 = box
            center_x, bottom_y = (x1 + x2) // 2, max(y1, y2)
            c = names[class_id]

            center_point = np.float32([[[center_x, bottom_y]]])
            center_point_bev = cv2.perspectiveTransform(center_point, H)
            cx_bev, cy_bev = center_point_bev[0][0]

            distance = math.sqrt(
                (cx_bev - bev_width // 2) ** 2 + (cy_bev - bev_height) ** 2
            )

            if c == "car" or c == "truck":
                rect_height = 36
                rect_width = 18
                if distance < WARNING_DISTANCE_THRESHOLD:
                    cv2.rectangle(
                        bird_eye_view,
                        (
                            int(cx_bev - (rect_width + 6) / 2),
                            int(cy_bev - (rect_height + 12) / 2),
                        ),
                        (
                            int(cx_bev + (rect_width + 6) / 2),
                            int(cy_bev + (rect_height + 12) / 2),
                        ),
                        RED,
                        -1,
                    )
                    last_warning_time = pygame.time.get_ticks()
                cv2.rectangle(
                    bird_eye_view,
                    (int(cx_bev - rect_width / 2), int(cy_bev - rect_height / 2)),
                    (int(cx_bev + rect_width / 2), int(cy_bev + rect_height / 2)),
                    YELLOW,
                    -1,
                )
                cv2.circle(radar_image, (int(cx_bev), int(cy_bev)), 5, YELLOW, -1)
            elif c == "motorcycle":
                rect_width = 10
                rect_height = 30
                if distance < WARNING_DISTANCE_THRESHOLD:
                    cv2.rectangle(
                        bird_eye_view,
                        (
                            int(cx_bev - (rect_width + 4) / 2),
                            int(cy_bev - (rect_height + 12) / 2),
                        ),
                        (
                            int(cx_bev + (rect_width + 4) / 2),
                            int(cy_bev + (rect_height + 12) / 2),
                        ),
                        RED,
                        -1,
                    )
                    last_warning_time = pygame.time.get_ticks()
                cv2.rectangle(
                    bird_eye_view,
                    (int(cx_bev - rect_width / 2), int(cy_bev - rect_height / 2)),
                    (int(cx_bev + rect_width / 2), int(cy_bev + rect_height / 2)),
                    PURPLE,
                    -1,
                )
                cv2.circle(radar_image, (int(cx_bev), int(cy_bev)), 5, PURPLE, -1)
            else:
                if distance < WARNING_DISTANCE_THRESHOLD:
                    cv2.circle(bird_eye_view, (int(cx_bev), int(cy_bev)), 12, RED, -1)
                    last_warning_time = pygame.time.get_ticks()
                cv2.circle(bird_eye_view, (int(cx_bev), int(cy_bev)), 10, PINK, -1)
                cv2.circle(radar_image, (int(cx_bev), int(cy_bev)), 5, PINK, -1)

            if show_labels:
                label_text = f"{c}"
                label_x = int(cx_bev) + 5  # 稍微偏移標籤位置
                label_y = int(cy_bev) - 5
                cv2.putText(
                    bird_eye_view,
                    label_text,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    WHITE,
                    1,
                )
                cv2.putText(
                    radar_image,
                    label_text,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    WHITE,
                    1,
                )

            if distance < WARNING_DISTANCE_THRESHOLD:
                cv2.putText(
                    bird_eye_view,
                    "WARNING!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    RED,
                    2,
                )
                cv2.putText(
                    radar_image,
                    "WARNING!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    RED,
                    2,
                )
                try:
                    if not warning_channel.get_busy():
                        warning_channel.play(warning_sound, maxtime=5000)
                except Exception as e:
                    print(f"Error playing sound: {e}")

            # 記錄數據到 CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            write_to_csv(
                [
                    timestamp,
                    track_id,
                    c,
                    cx_bev,
                    cy_bev,
                    distance,
                ]
            )

    cv2.rectangle(
        bird_eye_view,
        (bev_width // 2 - 10, bev_height),
        (bev_width // 2 + 10, bev_height + 30),
        GREEN,
        -1,
    )
    cv2.rectangle(
        radar_image,
        (bev_image_width // 2 - 10, bev_image_height),
        (bev_image_width // 2 + 10, bev_image_height + 30),
        GREEN,
        -1,
    )
    for point in src_points:
        cv2.circle(annotated_frame, (int(point[0]), int(point[1])), 5, GREEN, -1)

    cv2.polylines(
        annotated_frame,
        [src_points.astype(int)],
        isClosed=True,
        color=GREEN,
        thickness=2,
    )

    cv2.imshow("RGB", annotated_frame)
    cv2.imshow("Radar Image", radar_image)
    cv2.imshow("Bird's Eye View Image", bird_eye_view)

    # Update dashboard
    dashboard.fill(BLACK)
    display_text(dashboard, f"Detected Objects: {detected_objects_count}", (10, 10))

    if pygame.time.get_ticks() - last_warning_time < 1000:  # Show warning for 1 second
        display_text(dashboard, "WARNING!", (10, 40), RED)

    # Display the toggle option for the grid
    display_text(
        dashboard,
        "",
        (10, dashboard_height - 30),
        WHITE,
    )

    # 更新滑塊和閾值
    pygame_widgets.update(events)
    WARNING_DISTANCE_THRESHOLD = warning_distance_slider.getValue()
    output_box.setText(f"Warning Distance: {WARNING_DISTANCE_THRESHOLD}")

    pygame.display.flip()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
