import cv2
import os

dir_names = os.listdir()[0:-1]
path = '../EE6222_frames/'
for dir_name in dir_names:
    print("Processing: %s" % dir_name)
    path_dir = os.path.join(path, dir_name)
    video_names = os.listdir(dir_name)
    for name in video_names:
        cap = cv2.VideoCapture(os.path.join(dir_name, name))
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
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
                print("Video %s Done" % name)
                break
            cv2.imwrite(video_path + '\img_{0:05d}.jpg'.format(i), frame)
            i += 1
        # 完成所有操作后，释放捕获器
        cap.release()
