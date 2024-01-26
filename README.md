## Effective Image Tampering Localization with Multi-Scale ConvNeXt Feature Fusion  

### Network Architecture
<center> <img src="fig/Fig-1.png" alt="architecture"/> </center>

### Prerequisites
See ./requirements.txt
And refer to [official MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/).

### Training
Prepare for the training dataset like VOCdevkit and move it into `./data/`.
The pre-trained model can be downloaded as following:[pre-trained model for backbone](https://pan.baidu.com/s/1CWkdVMwPmgQnKVQZNM1f9g), code `0yh3`.
```python
python train.py
```

### Testing

Download the [weights](https://pan.baidu.com/s/1AI2KQJmBdEeGtPGZGyNkWQ)(code `jjyl`) and move it into the `./checkpoints/`.
To run all images in "samples/" directory, run:
```
python test.py
```

## Bibtex
 ```
@article{zhu2024effective,
  title={Effective image tampering localization with multi-scale convnext feature fusion},
  author={Zhu, Haochen and Cao, Gang and Zhao, Mo and Tian, Huawei and Lin, Weiguo},
  journal={Journal of Visual Communication and Image Representation},
  volume={98},
  pages={103981},
  year={2024},
  publisher={Elsevier}
}
```
### Contact

If you have any questions, please contact me(zhuhc_98@163.com).
