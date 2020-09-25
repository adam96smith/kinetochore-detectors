# kinetochore detector 

The following documentation explains the detectors for 3D time-lapse data of kinetochores. This explains how to use the tensorflow codes for both single-slice and multi-slice detectors.

## Setup

### Prerequisites
- Tensorflow v2.0.0

### Recommended
- Linux with Tensorflow GPU edition + cuDNN

### Getting Started

Example for training and using the single-slice Newby model:

```sh
# Traing the model
python newby.py \
  --mode train \
  --epochs 200 \
  --checkpoint newby_cp
# Use the model on real data. (real data in a dir called 'test')
python pix2pix.py \
  --mode evaluate \
  --test_dir test \
  --out_dir out_test \
  --checkpoint newby_cp
```

Requirements:

The data must be 4-D 32-bit floating point arrays with x,y,z dimensions before the time dimension.

Default is not set to saving the outputs, the evaluate() function has it commented out at the end.
