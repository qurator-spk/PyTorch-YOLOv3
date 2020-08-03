from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import click
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable

import pandas as pd


@click.command()
@click.argument('image_folder', type=click.Path(exists=True))
@click.argument('model_def', type=click.Path(exists=True))
@click.argument('weights_path', type=click.Path(exists=True))
@click.argument('result_file', type=click.Path())
# @click.argument('class_path', type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=128, help="size of the batches")
@click.option("--img-size", type=int, default=416, help="size of each image dimension")
@click.option('--n-cpu', type=int, default=4, help="number of cpu threads to use during batch generation")
@click.option("--conf-thres", type=float, default=0.8, help="object confidence threshold")
@click.option("--nms-thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
@click.option('--perform-nms', type=bool, is_flag=True, default=False,
              help="add if you actually want non-maximum supression")
def cli(image_folder, model_def, weights_path, result_file, batch_size, img_size, n_cpu,
        conf_thres, nms_thres, perform_nms):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )

    # noinspection PyUnresolvedReferences
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    results = []

    print("\nPerforming object detection:")
    for batch_i, (img_paths, input_imgs) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Configure input
        input_ten = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            imgs_detections = model(input_ten)

            if perform_nms:
                imgs_detections = non_max_suppression(imgs_detections, conf_thres, nms_thres)

        for (path, img, detections) in enumerate(zip(img_paths, input_imgs, imgs_detections)):

            # img = np.array(Image.open(path))

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, img_size, img.shape[:2])

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    box_w = x2 - x1
                    box_h = y2 - y1

                    results.append((path, x1, y1, box_w, box_h, conf, cls_conf, cls_pred))

        del input_imgs
        del input_ten
        del imgs_detections

        # Save image and detections
        # imgs.extend(img_paths)
        # img_detections.extend(detections)

    result = pd.DataFrame(results, columns=['path', 'x', 'y', 'width', 'height', 'conf', 'cls_conf', 'cls_pred'])

    result.to_pickle(result_file)
