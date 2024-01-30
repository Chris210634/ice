import torch
from PIL import Image
import os, sys
sys.path.append(os.getcwd())
from argparse_parameters import get_arg_parser
from source.trainer import *
from source.transforms import *
from source.samplers import *
from source.utils import *
from scripts.make_caption_utils import get_image_name_list
import argparse
import open_clip

parser = get_arg_parser()
parser.add_argument('--prompt', default = "What is in this image? ", type=str)
parser.add_argument('--suffix', default = "", type=str)
args = parser.parse_args()
print(args)

###############################################################################

def _generate(image_files, caption_model, prompt, tokenizer):
    ''' batched. Expect image_files to be list of str image paths. '''
    
    def _tokenize(x, tokenizer):
        x_tokenized = tokenizer(x).squeeze()
        start_token = 49406
        end_token = 49407
        assert x_tokenized[0] == start_token
        return x_tokenized[:list(x_tokenized).index(end_token)]
    
    def _generate_macro(caption_model, im, prompt):
        text=torch.ones((im.shape[0], 1), device='cuda', dtype=torch.long)*prompt
        generated = caption_model.generate(
                    im.cuda(), 
                    text=text,
                    generation_type='top_p')
        return generated
    
    caption_model.eval()
    
    outputs = []
    prompt_extended = _tokenize(prompt, tokenizer).cuda()
    
    test_xform = get_test_transform()
    images = []
    for imagefile in image_files:
        raw_image = Image.open(imagefile).convert("RGB")
        images.append(test_xform(raw_image))
    
    generated = _generate_macro(
        caption_model, 
        torch.stack(images).cuda(), 
        prompt_extended)
    
    assert len(generated) == len(image_files)
    for i in range(len(generated)):
        outputs.append(open_clip.decode(generated[i]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
    return outputs

###############################################################################

tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
caption_model, _, _ = open_clip.create_model_and_transforms(
      model_name="coca_ViT-L-14",
      pretrained="laion2B-s13B-b90k",
      cache_dir=args.cache_dir
    )
caption_model = caption_model.cuda()

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
        outputs = _generate(
            image_name_list[begin:end], 
            caption_model, 
            args.prompt,
            tokenizer
        )
        for o in outputs:
            answer_list.append(o.replace('\n', ' '))
            if begin == 0:
                print(o.replace('\n', ' '))
        ptr = end
        pbar.update(end-begin)
        
assert len(answer_list) == len(image_name_list)
with open('captions/COCA_captions/captions_{}{}'.format(dataset, args.suffix), 'w') as g:
    for imagename, answer in zip(image_name_list, answer_list): 
        assert imagename[:len(args.data_dir)] == args.data_dir
        g.write(imagename[len(args.data_dir):] + '<sep>' + answer + '<eol>')
        g.write('\n')