# ICE: Image Caption Encoding 🧊
-----------------------------------------------------

**Authors contributed equally to this repository, the comit history does not reflect contributions.**

Code in this repo uses code from [multimodal prompt learning](https://github.com/muzairkhattak/multimodal-prompt-learning), which in turn uses code from [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp).

## ⏳ Installation
-------------------

* Install dassl library and other requirements.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
pip install -r requirements.txt
python setup.py develop
cd ..

pip install open_clip_torch
pip install pytorch_metric_learning
```

* Create a directory somewhere called `data/`. Download all 15 zip files from [this shared Google Drive](https://drive.google.com/drive/folders/1kvh5VG4ruGOcSiHKJX9dWJhPAGVgPSZs?usp=drive_link) and unzip them into `data/`. The resulting file tree should look like:
```
data/
|-- caltech-101
|-- dtd
|-- eurosat
|-- fgvc_aircraft
|-- food-101
|-- imagenet
|-- imagenet-adversarial
|-- imagenet-rendition
|-- imagenet-sketch
|-- imagenetv2
|-- oxford_flowers
|-- oxford_pets
|-- stanford_cars
|-- sun397
|-- ucf101
```

Alternatively, follow the download instructions here (some dataset links are stale; may also need to reorganize the directory structure):
[installing datasets](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/DATASETS.md)

Modify the following two lines in `argparse_parameters.py` to reflect where you have your `data/` dir and where you want the pretrained CLIP weights to be cached (which could be many gigabytes)

```python
parser.add_argument('--cache_dir', default = "", type =str) # set to directory where you want large pretrained model weights to be cached
parser.add_argument('--data_dir', default = "", type =str)  # set to parent directory of data/
```

## 🧪 Experiements
---------------------------

### (1) Generate Captions 
ICE works with any captioner or VLM. We tried CoCa, BLIP-2, and LLAVA. We provide three example captions for each image in each dataset in the `captions/` folder. These can be used directrly. However, if you want to generate them yourself, you can run:

**Run:** `sh run_make_captions.sh`

The above script generates three captions per image per dataset, for CoCa, BLIP-2 and LLAVA. 

For LLAVA, the conda environment should be setup according to their github repo: [here](https://github.com/haotian-liu/LLaVA)


### (2) Run ICE with Generated Captions

**Run:** `sh run_ice.sh`


##  🧊 Results
---------------------------

See `ICE_results.xlsx`.

## 🧪 Ablation Experiements
---------------------------

| Ablation Description | Command to run |
| -------------------- | -------------- |
| Caption features zero-shot | `scripts/run_caption_features_zs.sh` |
| Image features zero-shot | `scripts/run_image_features_zs.sh` |
| Varying lambdas | `scripts/run_varying_lam.sh` |
| ICE with different captioner-image encoder combos | `run_scaling_law.sh` |


