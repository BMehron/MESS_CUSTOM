import tqdm
import os
from pathlib import Path
import gdown

import numpy as np
from PIL import Image

# Inria dataset color to class mapping
COLOR_TO_CLASS = {0: [0, 0, 0],  #  not building
                  1: [255, 255, 255],  # building
                }

VAL_SET = [
    'austin1',
    'austin2',
    'austin3',
    'austin4',
    'austin5',
    'chicago1',
    'chicago2',
    'chicago3',
    'chicago4',
    'chicago5',
    'kitsap1',
    'kitsap2',
    'kitsap3',
    'kitsap4',
    'kitsap5',
    'tyrol-w1',
    'tyrol-w2',
    'tyrol-w3',
    'tyrol-w4',
    'tyrol-w5',
    'vienna1',
    'vienna2',
    'vienna3',
    'vienna4',
    'vienna5', 
]

H_SIZE = 640
W_SIZE = 640

def download_dataset(dataset_dir):
    """
    Downloads the dataset
    """
    # TODO: Add an automated script if possible, otherwise remove code
    print('Downloading dataset...')
    # Downloading zip
    os.system('cd ' + str(dataset_dir))
    os.system('curl -k https://files.inria.fr/aerialimagelabeling/getAerial.sh | bash')
    os.system('mv AerialImageDataset Inria')
    for i in range(1, 6):
        os.system(f'rm aerialimagelabeling.7z.00{i}')
    os.system('rm NEW2-AerialImageDataset.zip')

def get_tiles(input, h_size=H_SIZE, w_size=W_SIZE, padding=0):
    input = np.array(input)
    h, w = input.shape[:2]
    tiles = []
    for i in range(0, h, h_size):
        for j in range(0, w, w_size):
            tile = input[i:i + h_size, j:j + w_size]
            if tile.shape[:2] == [h_size, w_size]:
                tiles.append(tile)
            else:
                # padding
                if len(tile.shape) == 2:
                    # Mask (2 channels, padding with ignore_value)
                    padded_tile = np.ones((h_size, w_size), dtype=np.uint8) * padding
                else:
                    # RGB (3 channels, padding usually 0)
                    padded_tile = np.ones((h_size, w_size, tile.shape[2]), dtype=np.uint8) * padding
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tiles.append(padded_tile)
    return tiles

def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'Inria'
    if not ds_path.exists():
        download_dataset(dataset_dir)
    else:
        print(f'Dataset was found. Tile size {H_SIZE} x {W_SIZE}')

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']: #Add 'train' if you want
        img_dir = ds_path / 'images_detectron2' / f'{split}_{H_SIZE}'
        anno_dir = ds_path / 'annotations_detectron2' / f'{split}_{W_SIZE}'
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        #number_of_images = sum(1 for _ in (ds_path / split / 'images').glob('*.tif'))

        for img_path in tqdm.tqdm((ds_path / split / 'images').glob('*.tif')):
            image_name = img_path.stem
            if split == 'train' and image_name not in VAL_SET:
                continue
                
            img = Image.open(img_path).convert('RGB')
            tiles = get_tiles(img, padding=0)
            for i, tile in enumerate(tiles):
                Image.fromarray(tile).save(img_dir / f'{image_name}_{i}.png')
                
            if split == 'train':
                # Open mask        
                mask = np.array(Image.open(ds_path / split / 'gt' / f'{image_name}.tif'))
    
                # Change building class idx by 1.
                mask[mask == 255] = 1
                tiles = get_tiles(mask, padding=255)
                for i, tile in enumerate(tiles):
                    Image.fromarray(tile).save(anno_dir / f'{image_name}_{i}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
