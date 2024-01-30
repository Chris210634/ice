import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
import argparse
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import open_clip

from source.utils import *
from source.losses import *
from source.samplers import *
from source.transforms import *
from source.models import *
from source.trainer import *
from argparse_parameters import get_arg_parser

# Ensembling three captions. These are the file suffixes
COCA_PROMPT_EXT = [
    ".a_photo_of",
    ".a",
    ".a_photo_containing"
]

BLIP_PROMPT_EXT = [
    "",
    ".concise",
    ".specific"
]

LLAVA_PROMPT_EXT = [
    "",
    ".concise",
    ".specific"
]

parser = get_arg_parser()
parser.add_argument('--subsample_classes', default = 'all', type=str)

##### ICE method #####
parser.add_argument('--ice', action='store_true')      # Whether to use ICE method
parser.add_argument('--ice_k', default = 5, type=int)  # K value for top-K
# A scalar to multiply with caption scores. 
# NOTE: this is multiplied with the normalized top-K standard deviation scores 
parser.add_argument('--ice_lambda', default = 0.08, type=float)  
# Whether we are just caching data for offline analysis, string is save dir 
parser.add_argument('--captioner', default='BLIP', type=str) # BLIP, LLAVA, COCA
parser.add_argument('--v', default = 3, type=int) # number of captions to average features over
parser.add_argument('--use_fixed_lambda', action='store_true') # overide adaptive lambda
parser.add_argument('--use_all_logits', action='store_true') # override top-K mechanism

args = parser.parse_args()
print(args)

if args.captioner == 'COCA':
    ICE_PROMPT_EXT = COCA_PROMPT_EXT
elif args.captioner == 'BLIP':
    ICE_PROMPT_EXT = BLIP_PROMPT_EXT
else:
    assert args.captioner == 'LLAVA'
    ICE_PROMPT_EXT = LLAVA_PROMPT_EXT
ICE_PROMPT_EXT = ICE_PROMPT_EXT[:args.v]
######################

device = "cuda"
modelname = args.modelname
pretrained = args.pretrained
cache_dir = args.cache_dir
d = args.d
n_classes = 1000 # place holder (dummy value)
tokenizer = open_clip.get_tokenizer(modelname)
model =  MyClip(modelname, pretrained, n_classes, d, 
                temp = args.temp, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=None,
                cache_dir=cache_dir)
model = model.cuda()
model.eval()
return_dict = {}
dataset_list = list(dataset_classes.keys())

for dataset in dataset_list:
    print('EVALUATING ON DATASET: ', dataset)
    cfg = argparse.Namespace()
    cfg.ROOT = args.data_dir
    cfg.NUM_SHOTS = 16
    cfg.SEED = 1
    cfg.SUBSAMPLE_CLASSES = 'all'
    dataset_class = dataset_classes[dataset]
    dset = dataset_class(cfg)
    classnames = dset.classnames
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

    # eval data loader
    dl_test = torch.utils.data.DataLoader(
                dset_test,
                num_workers=8,
                batch_size=32,
                pin_memory=True,
                drop_last=False,
                shuffle=False
    )

    ### Handle getting image features
    fn = 'cache/image_features.y_truth.{}{}{}.tup'.format(dataset, modelname, pretrained)
    img_list = dl_test.dataset.imgs
    if fn.split('/')[-1] in os.listdir('cache'):
        # if image features cached already, just use them
        image_features, y_truth, img_list_loaded = torch.load(fn)
        print('loaded image features from {}'.format(fn))
        # check that data list order is correct
        assert len(img_list) == len(img_list_loaded)
        assert all([img_list[i] == img_list_loaded[i] for i in range(len(img_list))]), 'cached image features bad, empty cache dir'
    else:
        # if image features no cached, then calculate them and cache the features
        with torch.no_grad():
            image_features, y_truth = get_features(dl_test, model, d=args.d)
        torch.save((image_features, y_truth, img_list), fn)
    image_features = F.normalize(image_features)
    
    ### Handle getting caption features  
    # Load in stored captions, get their embeddings, 
    # and compute the centroid of all caption features
    caption_features = []
    for ice_prompt_ext in ICE_PROMPT_EXT:
        caption_filename = 'captions/{}_captions/captions_{}{}'.format(
            args.captioner, dataset, ice_prompt_ext)
        print('pushing captions through encoder')
        caption_features.append(
            get_captions_from_file(caption_filename, dl_test, tokenizer, model, args).cpu()
        )
    caption_features = torch.stack(caption_features, dim=0).mean(0).float().cuda()
    caption_features = F.normalize(caption_features)
    
    # MAIN ALGORITHM
    def _evaluate(image_features, caption_features, 
                  text_features, y_truth, ice_lambda=args.ice_lambda):
        if args.ice:
            image_features = image_features.cuda()
            caption_features = caption_features.cuda()
            text_features = text_features.cuda()

            ##### ICE method #####
            # Obtain image and caption logits wrt. class embeddings
            i_logits = F.normalize(image_features) @ F.normalize(text_features).T
            c_logits = F.normalize(caption_features) @ F.normalize(text_features).T

            # Softmax the image and caption logits
            i_logits = F.softmax(i_logits, dim=1)
            c_logits = F.softmax(c_logits, dim=1)

            if args.use_all_logits:
                image_top_scores = i_logits
                caption_top_scores = c_logits
            else:
                # Precompute the top-K values and class indices 
                # for image and caption predictions
                top_k_preds = i_logits.topk(args.ice_k, dim=1).indices
                top_k_vals = i_logits.topk(args.ice_k, dim=1).values
                top_k_c_vals = c_logits.topk(args.ice_k, dim=1).values

                # For each prediction in the top-K image predictions, 
                # obtain the corresponding image and caption logits
                image_top_scores = torch.zeros(i_logits.shape[0], args.ice_k)
                caption_top_scores = torch.zeros(i_logits.shape[0], args.ice_k)
                for j in range(args.ice_k):
                    top_j_preds = top_k_preds[:, j]
                    j_scores = i_logits[torch.arange(i_logits.shape[0]), top_j_preds]
                    image_top_scores[:, j] = j_scores

                    j_scores = c_logits[torch.arange(c_logits.shape[0]), top_j_preds]
                    caption_top_scores[:, j] = j_scores

            if args.use_fixed_lambda:
                coef = args.ice_lambda
                ice_scores = (1. - coef) * image_top_scores.cuda() + coef * caption_top_scores.cuda()
                
            else:
                # Compute the ICE coefficient as \lambda * normalize(std(top-K image scores))
                # We find this to perform the best empirically
                coef = args.ice_lambda * F.normalize(
                    torch.stack(
                        (top_k_vals.std(1), top_k_c_vals.std(1))
                        , dim=1
                    )
                    , dim=1
                ).cuda()
                coef = coef[:, 1][:, None].repeat(1, caption_top_scores.shape[1])

                # Sum the image and caption scores to obtain the ICE scores
                ice_scores = image_top_scores.cuda() + coef * caption_top_scores.cuda()
            
            if args.use_all_logits:
                preds = ice_scores.argmax(1).cuda()
            else:
                ice_inds = ice_scores.argmax(1).cuda()
                preds = top_k_preds[torch.arange(top_k_preds.shape[0]), ice_inds]
                
            acc = (preds == y_truth.cuda()).float().mean().item() * 100.
            print('ice acc: ', acc)
            return acc
        
        else:
            with torch.no_grad():
                assert image_features.shape[0] == y_truth.shape[0]
                y_truth = y_truth.cuda()

                probs = caption_features @ F.normalize(text_features).T ###
                y_hat = probs.max(1).indices

            acc = (y_hat == y_truth).float().mean().item() * 100.
            print('acc: ', acc)
            return acc

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        prompt = prompt_strings[dataset]
        text = tokenizer(get_text_labels(dset.classnames, prompt))
        text_features = F.normalize(model.encode_text(text.cuda()).float())
        
    _acc = _evaluate(image_features, caption_features, 
                     text_features, y_truth,
                     ice_lambda = args.ice_lambda
                    )
    return_dict[dataset] = [_acc]

print('Results:')
print('', end=',')
for dataset in dataset_list:
    print(dataset, end=',')
print()
metrics = ['{} {} {} ZS'.format(args.modelname, args.pretrained, args.captioner)]
for i, metric in enumerate(metrics):
    print(metric, end=',')
    for dataset in dataset_list:
        print(return_dict[dataset][i], end=',')
    print()
