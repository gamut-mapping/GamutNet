import time
from argparse import ArgumentParser
from pathlib import Path
import random

random.seed(2021)

def main(args):
    split_output = Path('./split_output')
    split_output.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input)
    target_dir = Path(args.target)
    img_names = list(target_dir.glob('*.png'))

    list_idx = list(range(len(img_names)))
    random.shuffle(list_idx)
    train_length = round(0.9*(len(img_names)))

    write_txt(img_names, args.input,list_idx[:train_length], split_output / "train-input.txt")
    write_txt(img_names, args.input,list_idx[train_length:], split_output / "val-input.txt")
    write_txt(img_names, args.target,list_idx[:train_length], split_output / "train-target.txt")
    write_txt(img_names, args.target,list_idx[train_length:], split_output / "val-target.txt")

def write_txt(img_names, img_dir, list_idx, filename):
    with open(filename, 'w') as file:
        for idx in list_idx:
            file.write(img_dir + "/" + img_names[idx].name + "\n")

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                    help="input sRGB dir")
    parser.add_argument("-t", "--target", type=str,
                    help="target ProPhoto dir")
    main(parser.parse_args())  # parse args and start training

    end_time = time.time()
    duration = end_time - start_time
    duration = round(duration/3600, 2)
    print(f'---- FINISHED in {duration} hours ----')