import os
import sys
import json
import datetime
import numpy as np
import xml.etree.ElementTree as ET
import skimage.draw
import colorsys
from log import global_logger

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

#################################################
# Initialize Logger
logger = global_logger("Dataset")
################################################


class DocumentConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "document"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2     #4 has been tried 

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + figure + formula

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.90
    
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"


class DocumentDataset(utils.Dataset):
    """Generates the document dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_document(self, dataset_dir, subset):
        """Generate the requested number of images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        logger.info("load_document is called")
        # Add classes
        self.add_class("document", 1, "text")
        self.add_class("document", 2, "number")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add images
        logger.info("Storing Image Info from XML file .........")
        for filename in os.listdir(dataset_dir):
            if not filename.endswith('.xml'):
                continue

            xml_path = os.path.join(dataset_dir, filename)
            image_path = xml_path[:-3] + "jpg"
            base_name = filename[:-4]

            root = ET.parse(xml_path).getroot()

            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)

            polygons = []
            logger.info(filename + "Image info is loading.........")
            for ob in root.findall("object"):
                xmin = int(float(ob.find("bndbox").find("xmin").text))
                ymin = int(float(ob.find("bndbox").find("ymin").text))
                xmax = int(float(ob.find("bndbox").find("xmax").text))
                ymax = int(float(ob.find("bndbox").find("ymax").text))

                #xx, yy = np.meshgrid([x for x in range(xmin, xmax + 1)], [y for y in range(ymin, ymin + 1)])
                cat = ob.find("name").text
                polygons.append(
                    {
                        "category": 1 if cat == 'text' else 2,
                        "all_points_x": [xmin, xmin, xmax, xmax],
                        "all_points_y": [ymin, ymax, ymax, ymin]
                    }
                )

            self.add_image(
                "document",
                image_id=base_name,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
        logger.info("Images info loading complete.........")

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        print(info)
        if info["source"] == "document":	
            return info["path"]
        else:    
            super(self.__class__).image_reference(image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        logger.info("Generate instance masks for an image .........")
        # If not a document dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "document":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        category_list = []

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # print(rr, cc)
            mask[rr, cc, i] = 1
            category_list.append(p["category"])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(category_list, dtype=np.int32)

def train(model, args, config):
    """Train the model."""
    # Training dataset.
    logger.info("Traning dataset is preparing.........")
    dataset_train = DocumentDataset()
    dataset_train.load_document(args.dataset, "train")
    dataset_train.prepare()
    logger.info("Traning dataset has prepared.........")

    # Validation dataset
    logger.info("Validation dataset is preparing.........")
    dataset_val = DocumentDataset()
    dataset_val.load_document(args.dataset, "val")
    dataset_val.prepare()
    logger.info("Validation dataset has prepared.........")

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    logger.info("Training network heads.........")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=500,
                layers='heads')
    logger.info("Training network heads complete.........")
