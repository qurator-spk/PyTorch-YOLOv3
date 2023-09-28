from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import click
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import pandas as pd
import sqlite3
import io

from PIL import Image


class SqliteDataset(Dataset):

    def __init__(self, sqlite_file, img_size, table_name=None):

        super(Dataset, self).__init__()

        self.img_size = img_size

        if table_name is None:
            self.table_name = "images"
        else:
            self.table_name = table_name

        self.conn = None
        self.sqlite_file = sqlite_file

        with sqlite3.connect(sqlite_file) as conn:

            self.length = conn.execute("SELECT count(*) FROM {}".format(self.table_name)).fetchone()[0]

        print("SQLITE Dataset of size {}.".format(self.length))

    def __getitem__(self, index):

        if self.conn is None:
            self.conn = sqlite3.connect(self.sqlite_file)

            self.conn.execute('pragma journal_mode=wal')

        result = self.conn.execute("SELECT filename, data, scale_factor FROM {} WHERE rowid=?".format(self.table_name),
                                   (index,)).fetchone()
        if result is not None:
            filename, data, scale_factor = result

            buffer = io.BytesIO(data)

            img = Image.open(buffer).convert('RGB')
        else:
            filename = "ERROR-dummy.jpg"

            scale_factor = 1.0

            print('Something went wrong on image {}.'.format(index))
            print('Providing dummy result ...')

            img = Image.new('RGB', (256, 256))

        img = transforms.ToTensor()(img)

        img, _ = self.pad_to_square(img, 0)

        img_size = np.array(img.shape)

        img_size[1:] = img_size[1:] / scale_factor

        img = resize(img, self.img_size)

        return filename, img, img_size

    def __len__(self):
        return self.length

    @staticmethod
    def pad_to_square(img, pad_value):
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

        return img, pad


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
@click.option("--sqlite-table-name", type=str, default=None, help="")
# @click.option("--nms-thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
# @click.option('--perform-nms', type=bool, is_flag=True, default=False,
#              help="add if you actually want non-maximum supression")
def cli(image_folder, model_def, weights_path, result_file, batch_size, img_size, n_cpu, conf_thres, sqlite_table_name):

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

    if os.path.isdir(image_folder):

        dataloader = DataLoader(
            ImageFolder(image_folder, img_size=img_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_cpu,
        )
    else:
        dataloader = DataLoader(
            SqliteDataset(image_folder, img_size=img_size, table_name=sqlite_table_name),
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

        detections = []
        for i, part in enumerate(np.split(tmp, len(tmp))):
            detections.append(pd.DataFrame(part[..., :5].squeeze(),
                                           columns=['center_x', 'center_y', 'box_w', 'box_h', 'conf'],
                                           index=len(part.squeeze()) * [i]))

        detections = pd.concat(detections)

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
                    rescale_boxes(dpart[['x1', 'y1', 'x2', 'y2', 'conf']].values,
                                  img_size, [dpart.im_width.unique(), dpart.im_height.unique()]),
                    index=len(dpart) * [img_paths[i]],
                    columns=['x1', 'y1', 'x2', 'y2', 'conf']) for i, dpart in detections.groupby('image')]).\
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
