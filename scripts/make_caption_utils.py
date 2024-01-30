import torch
import argparse
from PIL import Image
from source.trainer import *
from source.transforms import *
from source.samplers import *
from source.utils import *


def get_image_name_list(dataset, args):
    print('Getting image path list for: ', dataset)
    cfg = argparse.Namespace()
    cfg.ROOT = args.data_dir
    cfg.NUM_SHOTS = 16
    cfg.SEED = 1
    cfg.SUBSAMPLE_CLASSES = 'all'
    dataset_class = dataset_classes[dataset]
    dset = dataset_class(cfg)
    test_xform = get_test_transform()

    if not dataset in ['ImageNet']:
        dset_test = dassl_dataset_conversion(dset, test_xform, 'test')
    else:
        # unlike the other datasets, when using immagenet,
        # following standard procedure, we use the 50,000 validation images
        # for testing
        dset_test = get_imagenet_val_dataset(
            test_xform,
            imagenet_root = os.path.join(args.data_dir, 'imagenet'),
            split=False)

    print('{} has {} test samples'.format(dataset, len(dset_test)))
    rl = [i[0] for i in dset_test.imgs]
    print('First image path: ', rl[0])
    return rl