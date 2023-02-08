import os
import time
from argparse import ArgumentParser
from pathlib import Path

from inference import NoPWInferenceAgent


def main(input_path, output_path, version_path, ckpt_filename):
    input_path = Path(input_path)
    if input_path.is_dir():
        input_files = [Path(entry.path) for entry in os.scandir(input_path)]
    elif input_path.is_file():
        input_files = [Path(line) for line in input_path.read_text().splitlines()]
    else:
        raise RuntimeError('input_path must be an existing directory.')
    assert all(input_file.is_file() for input_file in input_files), 'all the input files must exist.'
    num_input_files = len(input_files)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    inference_agent = NoPWInferenceAgent(version_path=version_path, ckpt_filename=ckpt_filename)
    for i, input_img_path in enumerate(input_files):
        print(f"Process {i + 1} / {num_input_files} image.")
        output_img_path = output_path / input_img_path.name
        inference_agent.single_image_inference_using_cnn(input_img_path, output_img_path)


# for dev and debug
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--version_path', type=str, required=True)
    parser.add_argument('--ckpt_filename', type=str, required=True)
    parser.add_argument('-i', '--input_path', type=str, required=True, help='either a directory or a text file')
    parser.add_argument('-o', '--output_path', type=str, required=True)
    args = parser.parse_args()

    started_at = time.time()
    main(**vars(args))
    print(f'---- FINISHED in {time.time() - started_at} seconds ----')
