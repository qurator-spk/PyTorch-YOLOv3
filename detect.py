from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import click
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import pandas as pd


@click.command()
@click.argument('image_folder', type=click.Path(exists=True))
@click.argument('model_def', type=click.Path(exists=True))
@click.argument('weights_path', type=click.Path(exists=True))
@click.argument('result_file', type=click.Path())
# @click.argument('class_path', type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=16, help="size of the batches")
@click.option("--img-size", type=int, default=416, help="size of each image dimension")
@click.option('--n-cpu', type=int, default=4, help="number of cpu threads to use during batch generation")
@click.option("--conf-thres", type=float, default=0.01, help="object confidence threshold")
# @click.option("--nms-thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
# @click.option('--perform-nms', type=bool, is_flag=True, default=False,
#              help="add if you actually want non-maximum supression")
def cli(image_folder, model_def, weights_path, result_file, batch_size, img_size, n_cpu, conf_thres):

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

    results = []

    print("\nPerforming object detection:")

    progress = tqdm(enumerate(dataloader), total=len(dataloader))

    num_found = 0
    check_point = 10000

    for batch_i, (img_paths, input_imgs, img_sizes) in progress:

        # Configure input
        input_ten = Variable(input_imgs.type(Tensor))

        img_sizes = img_sizes.cpu().numpy()

        # Get detections
        with torch.no_grad():
            output = model(input_ten)

            tmp = output.cpu().numpy()

        detections = \
            pd.concat(
                [pd.DataFrame(p.squeeze(), columns=['center_x', 'center_y', 'box_w', 'box_h', 'conf', 'score'],
                              index=len(p.squeeze()) * [i])
                 for i, p in enumerate(np.split(tmp, len(tmp)))]
            )

        detections['x1'] = detections.center_x - detections.box_w / 2.0
        detections['y1'] = detections.center_y - detections.box_h / 2.0
        detections['x2'] = detections.center_x + detections.box_w / 2.0
        detections['y2'] = detections.center_y + detections.box_h / 2.0

        detections = detections.reset_index().rename(columns={'index': 'image'})

        detections['im_width'] = img_sizes[detections.image, 1]
        detections['im_height'] = img_sizes[detections.image, 2]

        detections = detections.loc[detections.conf > conf_thres]

        if len(detections) == 0:
            continue

        detections =\
            pd.concat(
                [pd.DataFrame(
                    rescale_boxes(dpart[['x1', 'y1', 'x2', 'y2', 'conf', 'score']].values,
                                  img_size, [dpart.im_width.unique(), dpart.im_height.unique()]),
                    index=len(dpart) * [img_paths[i]],
                    columns=['x1', 'y1', 'x2', 'y2', 'conf', 'score']) for i, dpart in detections.groupby('image')]).\
                reset_index().rename(columns={'index': 'path'})

        detections['box_w'] = detections.x2 - detections.x1
        detections['box_h'] = detections.y2 - detections.y1

        num_found += len(detections)

        progress.set_description("Num found: {}".format(num_found))

        results.append(detections[['path', 'x1', 'y1', 'box_w', 'box_h', 'conf']])

        del input_imgs
        del input_ten
        del output

        if num_found > check_point:
            pd.concat(results).to_pickle(result_file)
            check_point = num_found + 10000

    pd.concat(results).to_pickle(result_file)
