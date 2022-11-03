import cv2
import os

video_root = 'datasets/ee6222/test'
video_names = os.listdir(video_root)
path = 'datasets/EE6222_frames_test/'

path_dir = path
for name in video_names:
    video_mp4_location = os.path.join(video_root, name)
    cap = cv2.VideoCapture(video_mp4_location)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    video_path = os.path.join(path_dir, name[:-4])
    i = 1
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    if not cap.isOpened():
        print(f"Cannot open camera for {video_mp4_location}")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Done: %s" % name, end="\r")
            break
        cv2.imwrite(video_path + '/img_{0:05d}.jpg'.format(i), frame)
        i += 1
    cap.release()
print('Extraction completed')
