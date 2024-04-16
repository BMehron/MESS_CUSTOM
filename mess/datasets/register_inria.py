
import os
from detectron2.utils.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap


CLASSES = [
    'other', 
    'building'
]

def load_sem_seg_without_gt(image_root, image_ext="jpg"):
    """
    
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (
            os.path.join(image_root, f)
            for f in PathManager.ls(image_root)
            if f.endswith(image_ext)
        ),
        key=lambda file_path: file2id(image_root, file_path),
    )

    dataset_dicts = []
    for img_path in input_files:
        record = {}
        record["file_name"] = img_path
        dataset_dicts.append(record)

    return dataset_dicts


# TODO: Add this script to mess/dataset/__init__.py
def register_dataset(root):
    ds_name = 'inria'
    root = os.path.join(root, 'Inria')

    for split, image_dirname, sem_seg_dirname, class_names in [
       #('val', 'images_detectron2/val', 'annotations_detectron2/val', CLASSES),
        ('test', 'images_detectron2/test_1024', '', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        org_images_dir = os.path.join(root, 'test/images')
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir: load_sem_seg_without_gt(
                x, image_ext='png'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            org_images_dir=org_images_dir,
            evaluator_type="inria_sem_seg",
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=[1],
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
