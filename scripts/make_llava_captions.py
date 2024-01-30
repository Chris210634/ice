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

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

parser = get_arg_parser()
parser.add_argument('--prompt', default = "What is in this image? ", type=str)
parser.add_argument('--suffix', default = "", type=str)
args = parser.parse_args()
print(args)

model_path = "liuhaotian/llava-v1.5-7b"

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def _generate(image_files, prompt):
    ''' batched. Expect image_files to be list of str image paths. '''

    model_path = "liuhaotian/llava-v1.5-7b"
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": '',
        "sep": ",",
        "temperature":0,
        "top_p":True,
        "num_beams": 1,
        "max_new_tokens":100
    })()

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    input_ids = input_ids.repeat(images_tensor.shape[0], 1)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )
    
    for o in outputs:
        o = o.strip()
        if o.endswith(stop_str):
            o = o[: -len(stop_str)]
        o = o.strip()
        
    return outputs

###############################################################################

disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)

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
with open('captions/LLAVA_captions/captions_{}{}'.format(dataset, args.suffix), 'w') as g:
    for imagename, answer in zip(image_name_list, answer_list):
        assert imagename[:len(args.data_dir)] == args.data_dir
        g.write(imagename[len(args.data_dir):] + '<sep>' + answer + '<eol>')
        g.write('\n')