# Lossy Image Compression using Autoencoders

This project demonstrates that we can use deep learning to compress images to very low bitrates and yet retain high qualities.This project was done as part of academic project for B.Tech degree by Abhishek Jha, Avik Banik, Soumitra Maity and Md. Akram Zaki of Kalyani Government Engineering College. This contains two already trained autoencoder and decoder model which you can train yourself if you would like. Currently it works only on PNG images.

## Installation

After cloning the git repo. Use the package manager [conda](https://www.anaconda.com/distribution/) to install the required dependencies for the.

```bash
conda env create -f environment.yml
```

## Usage

### Compressing

Run encode.py. This script will convert all the images in one directory and store the compressed version to out directory.
```python
usage: encode.py [-h] [--model [MODEL]] [--image [IMAGE]] [--out [OUT]]

optional arguments:
  -h, --help       show this help message and exit
  --model [MODEL]  Path for model checkpoint file [default: ./out/main.tar]
  --image [IMAGE]  Directory which holds the images to be compressed [default:
                   ./dataset/]
  --out [OUT]      Directory which will hold the compressed images [default:
                   ./out/compressed/]
```
sample:
```bash
python encode.py --image ../../someFolderContainingImages --out ../someFolder
```


### Decompressing the files

Run decode.py with the directory containing compressed files as parameter. The decoded files are saved to `out/decompressed`.
```bash
usage: decode.py [-h] [--model [MODEL]] [--compressed [COMPRESSED]]
                 [--out [OUT]]

optional arguments:
  -h, --help            show this help message and exit
  --model [MODEL]       Path for model checkpoint file [default:
                        ./out/main.tar]
  --compressed [COMPRESSED]
                        Directory which holds the compressed files [default:
                        ./out/compressed/]
  --out [OUT]           Directory which will hold the decompressed images
                        [default: ./out/decompressed/]
```

sample:
```bash
python decode.py --compressed ../../someFolderContainingFiles
```

### Training

Keep all the images in dataset folder and saves the training data for future training.For resuming training enter the checkpoint path parameter.
```bash
usage: train.py [-h] [--dataset-path [DATASET_PATH]]
                [--checkpoint-path [CHECKPOINT_PATH]] [--stop-at [STOP_AT]]
                [--save-at [SAVE_AT]]

optional arguments:
  -h, --help            show this help message and exit
  --dataset-path [DATASET_PATH]
                        Root directory of Images
  --checkpoint-path [CHECKPOINT_PATH]
                        Use to resume training from last checkpoint
  --stop-at [STOP_AT]   Epoch after you want to end training
  --save-at [SAVE_AT]   Directory where training state will be saved
```

sample:
```bash
python train.py --dataset-path ../input/ --stop-at 30 --save-at ./
```
## Switching models
We have created two models offering different compresison. You can use git to change the model.
```bash
git checkout model2 //for using model2
git checkout master //for using model1
```

## Testing
```bash
python test.py
```

Please make sure to keep original images in `dataset` folder and decompressed images in `out/decompressed` before running test.py. Or keep some images in `dataset` and run `encode.py` followed by `decode.py` with no parameters.

## Contribution
You are welcome to improve the project by adding or improving features. Just create the issue or pull request and I will get in touch. 
## License
[MIT](https://choosealicense.com/licenses/mit/)