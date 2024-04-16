
# Extends the detectron2.evaluation.SemSegEvaluator by an additional evaluation subset (classes_of_interest)
# Additionally reports CoI-mIoU as well as specificity and sensitivity for datasets with two classes

# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
import scipy
from collections import OrderedDict, defaultdict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

# setting path
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from prepare_datasets import prepare_inria

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import SemSegEvaluator
from PIL import Image

from typing import Optional, Union


################################## ADDED BY MEKHRON ############################

COLOR_TO_CLASS = {0: [0, 0, 0],  #  not building
                  1: [255, 255, 255],  # building
                }

# Copyright (c) Facebook, Inc. and its affiliates.


_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class InriaSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )
        self.org_images_dir = meta.org_images_dir
        if hasattr(meta, 'classes_of_interest'):
            self.classes_of_interest = np.bincount(meta.classes_of_interest, minlength=self._num_classes).astype(bool)
        else:
            self.classes_of_interest = np.ones(self._num_classes).astype(bool)

    @staticmethod
    def parse_name(file_name):
        split_idx = file_name.rfind('_')
        image_name = file_name[:split_idx]
        tile_number = int(file_name[split_idx+1:-4]) #remove .png
        return image_name, tile_number
        
    @staticmethod
    def label2rgb(mask):
        h, w = mask.shape[0], mask.shape[1]
        mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        mask_convert = mask[np.newaxis, :, :]
        mask_rgb[np.all(mask_convert == 0, axis=0)] = COLOR_TO_CLASS[0]
        mask_rgb[np.all(mask_convert == 1, axis=0)] = COLOR_TO_CLASS[1]
        return mask_rgb

    @staticmethod
    def connect_tiles(h_img, w_img, tiles, h_size=prepare_inria.H_SIZE, w_size=prepare_inria.W_SIZE):
        """_summary_

        Args:
            image_name (_type_): _description_
            h_size (_type_, optional): _description_. Defaults to prepare_inria.H_SIZE.
            w_size (_type_, optional): _description_. Defaults to prepare_inria.W_SIZE.

        Raises:
            Exception: _description_
        """
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        tile_idx = 0
        for i in range(0, h_img, h_size):
            for j in range(0, w_img, w_size):
                if tiles[tile_idx][0] != tile_idx:
                    raise Exception(f'Tile index does not match. Current index: {tiles[tile_idx][0]}, true index: {tile_idx}')
                # take care of border effect
                h_mask, w_mask = mask[i:i + h_size, j:j + w_size].shape
                mask[i:i + h_size, j:j + w_size] = tiles[tile_idx][1][:h_mask, :w_mask]                
                tile_idx += 1
        return mask

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = defaultdict(list)
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            prediction = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy().astype(dtype=np.uint8)
            image_name, tile_number = InriaSemSegEvaluator.parse_name(input["file_name"])
            self._predictions[image_name].append((tile_number, prediction))
    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean intersection-over-union averaged across classes of interest (CoI-mIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            #self._predictions = dict(all_gather(self._predictions))
            #self._predictions = dict(itertools.chain(*self._predictions))
            if not is_main_process():
                return

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            PathManager.mkdirs(os.path.join(self._output_dir,  'vis_predictions'))
            PathManager.mkdirs(os.path.join(self._output_dir,  'test_predictions'))
            self._predictions = dict(self._predictions)
            with PathManager.open(os.path.join(self._output_dir, "sem_seg_predictions.json"), "w") as f:
                f.write(json.dumps(self._predictions, cls=NumpyEncoder))
            for file_name in self._predictions:
                self._predictions[file_name].sort()
                image_name = file_name.split('/')[-1]
                #image = np.array(Image.open(os.path.join(self.org_images_dir, f'{image_name}.tif')).convert('RGB'))
                mask = InriaSemSegEvaluator.connect_tiles(5000, 5000, self._predictions[file_name])
                # Save Visualisation 
                # split_bar = np.zeros((image.shape[0], 50, image.shape[2]))
                # concat = np.concatenate((image, split_bar, InriaSemSegEvaluator.label2rgb(mask)), axis=1)
                #cv2.imwrite(os.path.join(self._output_dir,  'vis_predictions', 'catseg_' + image_name + '.png'), concat[:, :, ::-1])
                
                # Save Mask 
                mask[mask == 1] = 255
                Image.fromarray(mask).save(os.path.join(self._output_dir, 'test_predictions', image_name + '.tif'))
        
        print(f'The evaluation has been completed successfully. Find all results in {self._output_dir}')
        return {}



