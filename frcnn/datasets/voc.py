#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/frcnn/datasets/voc.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PASCAL Visual Object Classes dataset loader. Datasets available at:
# http://host.robots.ox.ac.uk/pascal/VOC/
#
# The dataset directory must contain the following sub-directories:
#
#   Annotations/
#   ImageSets/
#   JPEGImages/ 
#
# Typically, VOC datasets are stored in a VOCdevkit/ directory and identified
# by year (e.g., VOC2007, VOC2012). So, e.g., the VOC2007 dataset directory
# path would be: VOCdevkit/VOC2007
#

from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from typing import List
from typing import Tuple

from .training_sample import Box
from .training_sample import TrainingSample
from . import image
from frcnn.models import anchors


class DatasetOG:
    """
    A VOC dataset iterator for a particular split (train, val, etc.)
    """

    num_classes = 21
    class_index_to_name = {
        0: "background",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor"
    }

    def __init__(self, split, dir="VOCdevkit/VOC2007", feature_pixels=16, augment=True, shuffle=True,
                 allow_difficult=False, cache=True):
        """
        Parameters
        ----------
        split : str
          Dataset split to load: train, val, or trainval.
        dir : str
          Root directory of dataset.
        feature_pixels : int
          Size of each cell in the Faster R-CNN feature map (i.e., VGG-16 feature
          extractor output) in image pixels. This is the separation distance
          between anchors.
        augment : bool
          Whether to randomly augment (horizontally flip) images during iteration
          with 50% probability.
        shuffle : bool
          Whether to shuffle the dataset each time it is iterated.
        allow_difficult : bool
          Whether to include ground truth boxes that are marked as "difficult".
        cache : bool
          Whether to training samples in memory after first being generated.
        """
        if not os.path.exists(dir):
            raise FileNotFoundError("Dataset directory does not exist: %s" % dir)
        self.split = split
        self._dir = dir
        self.class_index_to_name = self._get_classes()
        self.class_name_to_index = {class_name: class_index for (class_index, class_name) in
                                    self.class_index_to_name.items()}
        self.num_classes = len(self.class_index_to_name)
        assert self.num_classes == Dataset.num_classes, "Dataset does not have the expected number of classes (found %d but expected %d)" % (
            self.num_classes, Dataset.num_classes)
        assert self.class_index_to_name == Dataset.class_index_to_name, "Dataset does not have the expected class mapping"
        self._filepaths = self._get_filepaths()
        self.num_samples = len(self._filepaths)
        self._gt_boxes_by_filepath = self._get_ground_truth_boxes(filepaths=self._filepaths,
                                                                  allow_difficult=allow_difficult)
        self._i = 0
        self._iterable_filepaths = self._filepaths.copy()
        self._feature_pixels = feature_pixels
        self._augment = augment
        self._shuffle = shuffle
        self._cache = cache
        self._unaugmented_cached_sample_by_filepath = {}
        self._augmented_cached_sample_by_filepath = {}

    def __iter__(self):
        self._i = 0
        if self._shuffle:
            random.shuffle(self._iterable_filepaths)
        return self

    def __next__(self):
        if self._i >= len(self._iterable_filepaths):
            raise StopIteration

        # Next file to load
        filepath = self._iterable_filepaths[self._i]
        self._i += 1

        # Augment?
        flip = random.randint(0, 1) != 0 if self._augment else 0
        cached_sample_by_filepath = self._augmented_cached_sample_by_filepath if flip else self._unaugmented_cached_sample_by_filepath

        # Load and, if caching, write back to cache
        if filepath in cached_sample_by_filepath:
            sample = cached_sample_by_filepath[filepath]
        else:
            sample = self._generate_training_sample(filepath=filepath, flip=flip)
        if self._cache:
            cached_sample_by_filepath[filepath] = sample

        # Return the sample
        return sample

    def _generate_training_sample(self, filepath, flip):
        # Load and preprocess the image
        scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url=filepath,
                                                                                         min_dimension_pixels=600,
                                                                                         horizontal_flip=flip)
        _, original_height, original_width = original_shape

        # Scale ground truth boxes to new image size
        scaled_gt_boxes = []
        for box in self._gt_boxes_by_filepath[filepath]:
            if flip:
                corners = np.array([
                    box.corners[0],
                    original_width - 1 - box.corners[3],
                    box.corners[2],
                    original_width - 1 - box.corners[1]
                ])
            else:
                corners = box.corners
            scaled_box = Box(
                class_index=box.class_index,
                class_name=box.class_name,
                corners=corners * scale_factor
            )
            scaled_gt_boxes.append(scaled_box)

        # Generate anchor maps and RPN truth map
        anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape=scaled_image_data.shape,
                                                                    feature_pixels=self._feature_pixels)
        gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(anchor_map=anchor_map,
                                                                                                anchor_valid_map=anchor_valid_map,
                                                                                                gt_boxes=scaled_gt_boxes)

        # Return sample
        return TrainingSample(
            anchor_map=anchor_map,
            anchor_valid_map=anchor_valid_map,
            gt_rpn_map=gt_rpn_map,
            gt_rpn_object_indices=gt_rpn_object_indices,
            gt_rpn_background_indices=gt_rpn_background_indices,
            gt_boxes=scaled_gt_boxes,
            image_data=scaled_image_data,
            image=scaled_image,
            filepath=filepath
        )

    def _get_classes(self):
        imageset_dir = os.path.join(self._dir, "ImageSets", "Main")
        classes = set(
            [os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_" + self.split + ".txt")])
        assert len(classes) > 0, "No classes found in ImageSets/Main for '%s' split" % self.split
        class_index_to_name = {(1 + v[0]): v[1] for v in enumerate(sorted(classes))}
        class_index_to_name[0] = "background"
        return class_index_to_name

    def _get_filepaths(self):
        image_list_file = os.path.join(self._dir, "ImageSets", "Main", self.split + ".txt")
        with open(image_list_file) as fp:
            basenames = [line.strip() for line in fp.readlines()]  # strip newlines
        image_paths = [os.path.join(self._dir, "JPEGImages", basename) + ".jpg" for basename in basenames]
        return image_paths
        # Debug: 60 car training images from VOC2012.
        image_paths = [
            "2008_000028",
            "2008_000074",
            "2008_000085",
            "2008_000105",
            "2008_000109",
            "2008_000143",
            "2008_000176",
            "2008_000185",
            "2008_000187",
            "2008_000189",
            "2008_000193",
            "2008_000199",
            "2008_000226",
            "2008_000237",
            "2008_000252",
            "2008_000260",
            "2008_000315",
            "2008_000346",
            "2008_000356",
            "2008_000399",
            "2008_000488",
            "2008_000531",
            "2008_000563",
            "2008_000583",
            "2008_000595",
            "2008_000613",
            "2008_000619",
            "2008_000719",
            "2008_000833",
            "2008_000944",
            "2008_000953",
            "2008_000959",
            "2008_000979",
            "2008_001018",
            "2008_001039",
            "2008_001042",
            "2008_001104",
            "2008_001169",
            "2008_001196",
            "2008_001208",
            "2008_001274",
            "2008_001329",
            "2008_001359",
            "2008_001375",
            "2008_001440",
            "2008_001446",
            "2008_001500",
            "2008_001533",
            "2008_001541",
            "2008_001631",
            "2008_001632",
            "2008_001716",
            "2008_001746",
            "2008_001860",
            "2008_001941",
            "2008_002062",
            "2008_002118",
            "2008_002197",
            "2008_002202",
            "2011_003247"
        ]
        return [os.path.join(self._dir, "JPEGImages", path) + ".jpg" for path in image_paths]

    def _get_ground_truth_boxes(self, filepaths, allow_difficult):
        gt_boxes_by_filepath = {}
        for filepath in filepaths:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            annotation_file = os.path.join(self._dir, "Annotations", basename) + ".xml"
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            assert tree != None, "Failed to parse %s" % annotation_file
            assert len(root.findall("size")) == 1
            size = root.find("size")
            assert len(size.findall("depth")) == 1
            depth = int(size.find("depth").text)
            assert depth == 3
            boxes = []
            for obj in root.findall("object"):
                assert len(obj.findall("name")) == 1
                assert len(obj.findall("bndbox")) == 1
                assert len(obj.findall("difficult")) == 1
                is_difficult = int(obj.find("difficult").text) != 0
                if is_difficult and not allow_difficult:
                    continue  # ignore difficult examples unless asked to include them
                class_name = obj.find("name").text
                bndbox = obj.find("bndbox")
                assert len(bndbox.findall("xmin")) == 1
                assert len(bndbox.findall("ymin")) == 1
                assert len(bndbox.findall("xmax")) == 1
                assert len(bndbox.findall("ymax")) == 1
                x_min = int(bndbox.find("xmin").text) - 1  # convert to 0-based pixel coordinates
                y_min = int(bndbox.find("ymin").text) - 1
                x_max = int(bndbox.find("xmax").text) - 1
                y_max = int(bndbox.find("ymax").text) - 1
                corners = np.array([y_min, x_min, y_max, x_max]).astype(np.float32)
                box = Box(class_index=self.class_name_to_index[class_name], class_name=class_name, corners=corners)
                boxes.append(box)
            assert len(boxes) > 0
            gt_boxes_by_filepath[filepath] = boxes
        return gt_boxes_by_filepath


class Dataset:
    """
    A VOC dataset iterator for a particular split (train, val, etc.)
    """

    num_classes = 15
    class_index_to_name = {
        0: 'background', 1: 'pieces', 2: 'bishop', 3: 'black-bishop', 4: 'black-king', 5: 'black-knight', 6: 'black-pawn',
        7: 'black-queen', 8: 'black-rook', 9: 'white-bishop', 10: 'white-king', 11: 'white-knight', 12: 'white-pawn',
        13: 'white-queen', 14: 'white-rook'
    }

    def __init__(self, split, dir="dataset/chess-pieces-dataset", feature_pixels=16, augment=True, shuffle=True,
                 allow_difficult=False, cache=True):
        """
        Parameters
        ----------
        split : str
          Dataset split to load: train, val, or trainval.
        dir : str
          Root directory of dataset.
        feature_pixels : int
          Size of each cell in the Faster R-CNN feature map (i.e., VGG-16 feature
          extractor output) in image pixels. This is the separation distance
          between anchors.
        augment : bool
          Whether to randomly augment (horizontally flip) images during iteration
          with 50% probability.
        shuffle : bool
          Whether to shuffle the dataset each time it is iterated.
        allow_difficult : bool
          Whether to include ground truth boxes that are marked as "difficult".
        cache : bool
          Whether to training samples in memory after first being generated.
        """
        if not os.path.exists(dir):
            raise FileNotFoundError("Dataset directory does not exist: %s" % dir)
        self.split = split
        self._dir = dir
        # self.class_index_to_name = self._get_classes()
        # self.class_index_to_name = self._add_background_class(self.class_index_to_name)
        self.class_name_to_index = {class_name: class_index for (class_index, class_name) in
                                    self.class_index_to_name.items()}
        self.num_classes = len(self.class_index_to_name)
        assert self.num_classes == Dataset.num_classes, "Dataset does not have the expected number of classes (found %d but expected %d)" % (
            self.num_classes, Dataset.num_classes)
        # assert self.class_index_to_name == Dataset.class_index_to_name, "Dataset does not have the expected class mapping"
        self._filepaths = self._get_filepaths()
        self.num_samples = len(self._filepaths)
        self._gt_boxes_by_filepath = self._get_ground_truth_boxes(filepaths=self._filepaths,
                                                                  allow_difficult=allow_difficult)
        self._i = 0
        self._iterable_filepaths = self._filepaths.copy()
        self._feature_pixels = feature_pixels
        self._augment = augment
        self._shuffle = shuffle
        self._cache = cache
        self._unaugmented_cached_sample_by_filepath = {}
        self._augmented_cached_sample_by_filepath = {}

    def __iter__(self):
        self._i = 0
        if self._shuffle:
            random.shuffle(self._iterable_filepaths)
        return self

    def __next__(self):
        if self._i >= len(self._iterable_filepaths):
            raise StopIteration

        # Next file to load
        filepath = self._iterable_filepaths[self._i]
        self._i += 1

        # Augment?
        flip = random.randint(0, 1) != 0 if self._augment else 0
        cached_sample_by_filepath = self._augmented_cached_sample_by_filepath if flip else self._unaugmented_cached_sample_by_filepath

        # Load and, if caching, write back to cache
        if filepath in cached_sample_by_filepath:
            sample = cached_sample_by_filepath[filepath]
        else:
            sample = self._generate_training_sample(filepath=filepath, flip=flip)
        if self._cache:
            cached_sample_by_filepath[filepath] = sample

        # Return the sample
        return sample

    def _generate_training_sample(self, filepath, flip):
        # Load and preprocess the image
        scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url=filepath,
                                                                                         min_dimension_pixels=600,
                                                                                         horizontal_flip=flip)
        _, original_height, original_width = original_shape

        # Scale ground truth boxes to new image size
        scaled_gt_boxes = []
        for box in self._gt_boxes_by_filepath[filepath]:
            if flip:
                corners = np.array([
                    box.corners[0],
                    original_width - 1 - box.corners[3],
                    box.corners[2],
                    original_width - 1 - box.corners[1]
                ])
            else:
                corners = box.corners
            scaled_box = Box(
                class_index=box.class_index,
                class_name=box.class_name,
                corners=corners * scale_factor
            )
            scaled_gt_boxes.append(scaled_box)

        # Generate anchor maps and RPN truth map
        anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape=scaled_image_data.shape,
                                                                    feature_pixels=self._feature_pixels)
        gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(anchor_map=anchor_map,
                                                                                                anchor_valid_map=anchor_valid_map,
                                                                                                gt_boxes=scaled_gt_boxes)

        # Return sample
        return TrainingSample(
            anchor_map=anchor_map,
            anchor_valid_map=anchor_valid_map,
            gt_rpn_map=gt_rpn_map,
            gt_rpn_object_indices=gt_rpn_object_indices,
            gt_rpn_background_indices=gt_rpn_background_indices,
            gt_boxes=scaled_gt_boxes,
            image_data=scaled_image_data,
            image=scaled_image,
            filepath=filepath
        )

    def _get_classes(self):
        imageset_dir = os.path.join(self._dir, "ImageSets", "Main")
        classes = set(
            [os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_" + self.split + ".txt")])
        assert len(classes) > 0, "No classes found in ImageSets/Main for '%s' split" % self.split
        class_index_to_name = {(1 + v[0]): v[1] for v in enumerate(sorted(classes))}
        class_index_to_name[0] = "background"
        return class_index_to_name

    def _add_background_class(self, class_index_to_name: dict):
        class_index_to_name = {(1 + k): v for k, v in class_index_to_name.items()}
        class_index_to_name[0] = "background"
        return class_index_to_name

    def _get_filepaths(self):
        image_paths = []
        for root, dirs, _ in os.walk(self._dir):
            for dir in dirs:
                if self.split == dir:
                    folder_path = os.path.join(root, dir)
                    for _, _, files in os.walk(folder_path):
                        for file in files:
                            if file.rsplit(".")[-1] == "jpg":
                                image_paths.append(os.path.join(root, self.split, file))
        return image_paths
        # Debug: 20 chess training images:
        image_paths = [
            '3bab0eaaeb63a2ac9ae4942df4006a25_jpg.rf.8fd1c7b01ae630cdb96546469e0c742d.jpg',
            '3bab0eaaeb63a2ac9ae4942df4006a25_jpg.rf.b78947d5207c15119ee81058a1b75c1e.jpg',
            '3161933dffedf8a859d6623a99492c53_jpg.rf.b3cca32040dcb031002296f83298f3d1.jpg',
            '254f92b18b2a81f88b85e7aed3cabc61_jpg.rf.a55e3d26992b9f4d43e7f317a078689b.jpg',
            'IMG_0291_JPG.rf.d2ba6353082aa25c15708824c08dfb27.jpg',
            '5758322233deed7ae7adc23536db2a4f_jpg.rf.469940331ca0c0fbabd2eaad8348ed71.jpg',
            '389b4c47568c78c44df11dbb1377ffea_jpg.rf.0185f6bf38d82f7cbf9365edd7b2bfc7.jpg',
            '4894f034a55eaa9252cd261a62b11d27_jpg.rf.e153d650cc91ee8985dbc0f9b5050e98.jpg',
            '4894f034a55eaa9252cd261a62b11d27_jpg.rf.bcd60bd54187dbd564c6d84e8a4d3cb9.jpg',
            'd079f4e77b2445abceca7534356db743_jpg.rf.e7fc6fdfea0d14dc4c82a6068b9e4159.jpg',
            '8ff64b3f770bfe96bdffc629efd16460_jpg.rf.7b4792b9f562b28d55342586be82fe91.jpg',
            'ddad9dc4d945006d66f5349d64498559_jpg.rf.48e7a4d1dbc55402801f6f3eb2515561.jpg',
            '1728cd731489df8bb8e0396e178fe393_jpg.rf.cf3127987c30548d691295953a2326db.jpg',
            '02f0931b536dfba10affc3231a3d64fb_jpg.rf.087fbe5ea178dd757f4eb065ae5cf941.jpg',
            '4894f034a55eaa9252cd261a62b11d27_jpg.rf.ec15ec6e91a0367ded74d29495beadca.jpg',
            'f041d3171dfe3137390c85fc5437e447_jpg.rf.3020fee02bc4def16c99bed406ad8671.jpg',
            '03886821377011fec599e8fa12d86e89_jpg.rf.7ec3f29be4f3793b35a2c4a9880d831c.jpg',
            '9146a6989dac08f1769e677064ebfb49_jpg.rf.f479d3177bde0b8beb172fcd798971f2.jpg',
            'a9768de3fceeeae2618f362870fb9a88_jpg.rf.444b950f0b329aa6e7ed17a86383606d.jpg',
            '673bcd0d44f495fbe9dd88d5cacfceb3_jpg.rf.3b647f8c3bb9f3fc64a0d0edf806f691.jpg',
            'IMG_0166_JPG.rf.866e83ca31acd30da2673fcb7e2abbfe.jpg',
        ]
        return [os.path.join(self._dir, self.split, path) for path in image_paths]

    def _get_ground_truth_boxes(self, filepaths, allow_difficult):
        gt_boxes_by_filepath = {}
        for filepath in filepaths:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            annotation_file = os.path.join(self._dir, self.split, basename) + ".xml"
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            assert tree != None, "Failed to parse %s" % annotation_file
            assert len(root.findall("size")) == 1
            size = root.find("size")
            assert len(size.findall("depth")) == 1
            depth = int(size.find("depth").text)
            assert depth == 3
            boxes = []
            for obj in root.findall("object"):
                assert len(obj.findall("name")) == 1
                assert len(obj.findall("bndbox")) == 1
                assert len(obj.findall("difficult")) == 1
                is_difficult = int(obj.find("difficult").text) != 0
                if is_difficult and not allow_difficult:
                    continue  # ignore difficult examples unless asked to include them
                class_name = obj.find("name").text
                bndbox = obj.find("bndbox")
                assert len(bndbox.findall("xmin")) == 1
                assert len(bndbox.findall("ymin")) == 1
                assert len(bndbox.findall("xmax")) == 1
                assert len(bndbox.findall("ymax")) == 1
                x_min = int(bndbox.find("xmin").text) - 1  # convert to 0-based pixel coordinates
                y_min = int(bndbox.find("ymin").text) - 1
                x_max = int(bndbox.find("xmax").text) - 1
                y_max = int(bndbox.find("ymax").text) - 1
                corners = np.array([y_min, x_min, y_max, x_max]).astype(np.float32)
                box = Box(class_index=self.class_name_to_index[class_name], class_name=class_name, corners=corners)
                boxes.append(box)
            if len(boxes) == 0:
                print(filepath)
            assert len(boxes) > 0
            gt_boxes_by_filepath[filepath] = boxes
        return gt_boxes_by_filepath
