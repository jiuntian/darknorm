import cv2
import os

dir_names = os.listdir('datasets/ee6222/train')
path = 'datasets/EE6222_frames/'
for dir_name in dir_names:
    print("Processing: %s" % dir_name)
    path_dir = os.path.join(path, dir_name)
    video_names = os.listdir(dir_name)
    for name in video_names:
        cap = cv2.VideoCapture(os.path.join(dir_name, name))
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
            ret, frame = cap.read()
            if not ret:
                print("Done: %s" % name, end="\r")
                break
            cv2.imwrite(video_path + '/img_{0:05d}.jpg'.format(i), frame)
            i += 1
        cap.release()
print('Extraction completed')
