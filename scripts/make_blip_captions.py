import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os, sys
sys.path.append(os.getcwd())
from argparse_parameters import get_arg_parser
from source.trainer import *
from source.transforms import *
from source.samplers import *
from source.utils import *
from scripts.make_caption_utils import get_image_name_list
import argparse

parser = get_arg_parser()
parser.add_argument('--prompt', default = "Question: What is in this image? Answer: A photo of ", type=str)
parser.add_argument('--suffix', default = "", type=str)
args = parser.parse_args()
print(args)

def _generate(image_files, prompt):
    ''' batched. Expect image_files to be list of str image paths. '''
    images = []
    for imagefile in image_files:
        raw_image = Image.open(imagefile).convert("RGB")
        images.append(vis_processors["eval"](raw_image))
    return model.generate({"image": torch.stack(images).cuda(), "prompt": prompt})

###############################################################################

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xxl", 
    is_eval=True, device=device)

dataset = args.dataset
image_name_list = get_image_name_list(dataset, args)
        
answer_list = []
device = 'cuda'
print('prompt:', args.prompt)

with tqdm(total=len(image_name_list)) as pbar:
    ptr = 0
    while ptr < len(image_name_list):
        begin = ptr
        end = min(ptr+args.bs, len(image_name_list))
        outputs = _generate(image_name_list[begin:end], args.prompt)
        for o in outputs:
            answer_list.append(o.replace('\n', ' '))
            if begin == 0:
                print(o.replace('\n', ' '))
        ptr = end
        pbar.update(end-begin)
        
assert len(answer_list) == len(image_name_list)
with open('captions/BLIP_captions/captions_{}{}'.format(dataset, args.suffix), 'w') as g:
    for imagename, answer in zip(image_name_list, answer_list): 
        assert imagename[:len(args.data_dir)] == args.data_dir
        g.write(imagename[len(args.data_dir):] + '<sep>' + 'a photo of ' + answer + '<eol>')
        g.write('\n')