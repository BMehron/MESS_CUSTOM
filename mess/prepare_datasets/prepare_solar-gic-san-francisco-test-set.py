from pathlib import Path
import os

# original images
mask_dir = Path('/home/data/solar-panel-segmentation/solar-gic-san-francisco-test-set/SegmentationClass')
image_dir = Path('/home/data/solar-panel-segmentation/solar-gic-san-francisco-test-set/JPEGImages')
# detectron dirs
img_dir = Path('/home/data/mess_datasets/solar-gic-san-francisco-test-set/images_detectron2')
anno_dir = Path('/home/data/mess_datasets/solar-gic-san-francisco-test-set/annotations_detectron2')

os.makedirs(img_dir, exist_ok=True)
os.makedirs(anno_dir, exist_ok=True)


def main():
    # Convert annotations to detectron2 format and symbolic link images.
    for mask_path in tqdm.tqdm(sorted((mask_dir).glob('*.png'))):
        file_name = str(mask_path).split('/')[-1]
        os.system(f'ln -s {image_dir / file_name} {img_dir / file_name}')
        mask = np.array(Image.open(mask_path))
        # Map RGB values to class index by applying a lookup table
        if len(mask.shape) == 3:
            mask[(mask == [250, 125, 187]).all(axis=-1)] = 1
            mask = mask[:, :, 0]
        Image.fromarray(mask).save(anno_dir / file_name)


if __name__ == '__main__':
    main()