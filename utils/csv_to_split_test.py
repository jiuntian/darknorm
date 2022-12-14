import os
import csv


def csv_deal(csv_file: str, csv_type):
    path = 'datasets/EE6222_frames_test/'
    with open(csv_file, newline="") as split_f:
        reader = csv.DictReader(split_f, delimiter="\t")
        save_txt = csv_type + '_' + "split1" + ".txt"
        with open(save_txt, 'w') as write_txt:
            for i, line in enumerate(reader):
                label = line["ClassID"]
                name = line["Video"][:-4]
                duration = str(len(os.listdir(path + name)))
                write_thing = name + ' ' + duration + ' ' + label + '\n'
                print("Writing：Video：" + name + 'Frames：' + duration + 'Label：' + label)
                write_txt.write(write_thing)


csv_deal("datasets/ee6222/test.txt", "test")
