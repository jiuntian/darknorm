import cv2
import os

# dir_name = os.listdir()
path = '/raid2/jiuntian/ee6222/darklight/datasets/EE6222_frames/'

path_dir = path
video_names = os.listdir()
for name in video_names:
    cap = cv2.VideoCapture(name)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    video_path = os.path.join(path_dir, name[:-4])
    i = 1
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("Done: %s" % name, end="\r")
            break
        cv2.imwrite(video_path + '/img_{0:05d}.jpg'.format(i), frame)
        i += 1
    # 完成所有操作后，释放捕获器
    cap.release()