import argparse
import os

import magic
from PIL import Image
from tqdm import tqdm


def check_is_tif(filepath: str) -> bool:
    allowed_types = [
        'image/tiff',
        'image/tif'
    ]

    if magic.from_file(filepath, mime=True) not in allowed_types:
        return False
    return True


def count_total(path: str) -> int:
    print('Please wait till total files are counted...')
    print(path)

    result = 0

    for root, _, files in os.walk(path):
        for name in files:
            if check_is_tif(os.path.join(root, name)) is True:
                result += 1

    return result


def convert(path) -> None:
    progress = tqdm(total=count_total(path))

    for root, _, files in os.walk(path):
        for name in files:
            if check_is_tif(os.path.join(root, name)) is True:
                file_path = os.path.join(root, name)
                outfile = os.path.splitext(file_path)[0] + ".png"
                
                try:
                    im = Image.open(file_path)
                    
                    im = im.convert('RGB')
                    im.thumbnail(im.size)
                    im.save(outfile, format='png')
                    os.unlink(file_path)
                except Exception as e:
                    print(e)

                progress.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recursive TIFF to JPEG converter')
    parser.add_argument('--path', type=str, help='Path do directory with TIFF files')
    args = parser.parse_args()
    
    convert(args.path)