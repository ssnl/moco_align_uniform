<!-- Copyright (c) 2020 Tongzhou Wang -->
# Momentum Contrast (MoCo) with Alignment and Uniformity Losses

This directory contains a PyTorch implementation of a MoCo variant using the Alignment and Uniformity losses proposed in paper: [**Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**](https://arxiv.org/abs/2005.10242):
```
@inproceedings{wang2020hypersphere,
  title={Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
  author={Wang, Tongzhou and Isola, Phillip},
  booktitle={International Conference on Machine Learning},
  organization={PMLR},
  pages={9929--9939},
  year={2020}
}
```

More code for this paper can be found at [this repository](https://github.com/SsnL/align_uniform).

---
**NOTE**

This README is for better equation readability under GitHub's new dark mode theme. If you are using the light mode, please use [this README](./README.md).

---

## Requirements
```
Python >= 3.6
torch >= 1.5.0
torchvision
```

## Datasets

### ImageNet

The full ImageNet compatible with PyTorch can be obtained online, e.g., by following instructions specified in [the official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet#requirements).

### ImageNet-100 Subset

The ImageNet-100 subset contains a randomly sampled 100 classes of the full ImageNet (1000 classes). The list of the 100 classes we used in our experiments are provided in [`scripts/imagenet100_classes.txt`](./scripts/imagenet100_classes.txt). This subset is identical to the one used in [Contrastive Multiview Coding (CMC)](https://arxiv.org/abs/1906.05849).

We provide a script that constructs proper symlinks to form the subset from the full ImageNet. You may invoke it as following:

```sh
python scripts/create_imagenet_subset.py [PATH_TO_EXISTING_IMAGENET] [PATH_TO_CREATE_SUBSET]
```

Optionally, you may add argument `--subset [PATH_TO_CLASS_SUBSET_FILE]` to specify a custom subset file, which should follow the same format as [`scripts/imagenet100_classes.txt`](./scripts/imagenet100_classes.txt). See [`scripts/create_imagenet_subset.py`](./scripts/create_imagenet_subset.py) for more options.

## Getting Started

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

+ To train a ResNet-50 encoder with loss <img src="https://latex.codecogs.com/svg.latex?\color{White}\begin{tabular}[b]{@{}c@{}}$3\cdot\mathcal{L}_\mathsf{align}(\alpha\mkern1.5mu{=}\mkern1.5mu2)+\mathcal{L}_\mathsf{uniform}(t\mkern1.5mu{=}\mkern1.5mu3)$\vspace{-5pt}\\{\color{black}.}\\\end{tabular} " align="middle" /> (default) on a 4-GPU machine, run:

  ```sh
  python main_moco.py \
      -a resnet50 \
      --lr 0.03 --batch-size 128 \
      --gpus 0 1 2 3 \
      --multiprocessing-distributed --world-size 1 --rank 0 \
      [PATH_TO_DATASET]
  ```

+ The following arguments control the loss form:
  <table>
    <tr>
      <th>Command-line Arguments</th>
      <th>Loss Term</th>
      <th>Default Values</th>
    </tr>
    <tr>
      <td><code>--moco-align-w AW --moco-align-alpha AALPHA</code></td>
      <td><img src="https://latex.codecogs.com/svg.latex?\color{White}\begin{tabular}[b]{@{}c@{}}$\texttt{AW}\cdot\mathcal{L}_\mathsf{align}(\alpha\mkern1.5mu{=}\mkern1.5mu\texttt{AALPHA})$\vspace{-8pt}\\{\color{black}.}\\\end{tabular}" align="middle" /></td>
      <td><code>AW=3 AALPHA=2</code></td>
    </tr>
    <tr>
      <td><code>--moco-unif-w UW --moco-unif-t UT</code></td>
      <td><img src="https://latex.codecogs.com/svg.latex?\color{White}\begin{tabular}[b]{@{}c@{}}$\texttt{UW}\cdot\mathcal{L}_\mathsf{uniform}(t\mkern1.5mu{=}\mkern1.5mu\texttt{UT})$\vspace{-8pt}\\{\color{black}.}\\\end{tabular}" align="middle" /></td>
      <td><code>UW=1 UT=3</code></td>
    </tr>
    <tr>
      <td><code>--moco-contr-w CW --moco-contr-tau CTAU</code></td>
      <td><img src="https://latex.codecogs.com/svg.latex?\begin{tabular}[b]{@{}c@{}}\color{White}$\texttt{CW}\cdot\mathcal{L}_\mathsf{contrastive}(\tau\mkern1.5mu{=}\mkern1.5mu\texttt{CTAU})$\vspace{-8pt}\\{\color{black}.}\\\end{tabular}" align="middle" /></td>
      <td><code>CW=0 CTAU=0.07</code></td>
    </tr>
  </table>

  ***Note***: By default, <img src="https://latex.codecogs.com/svg.latex?\color{White}\begin{tabular}[b]{@{}c@{}}{\color{black}.}\vspace{-4pt}\\$\mathcal{L}_\mathsf{uniform}$\end{tabular} " align="top" /> uses the "intra-batch" version, where the negative pair distances include both the distance between samples in each batch and features in queue, as well as pairwise distances within each batch ([Equation 18](https://arxiv.org/pdf/2005.10242.pdf#page=23)). The command-line flag `--moco-unif-no-intra-batch` switches to the form without using pairwise distances within batch ([Equation 17](https://arxiv.org/pdf/2005.10242.pdf#page=23)).

+ This repository also includes several techniques MoCo v2 added. To include those, set `--aug-plus --mlp --cos`, which turns on stronger augmentation, MLP header, and cosine learning rate scheduling.

+ For the ImageNet-100 subset, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677), with `--lr 0.03` per `--batch-size 128`. For other datasets (e.g., the full ImageNet), you may need to use other learning rate and batch size settings.

### Linear Classification

To evaluate an encoder by fitting a supervised linear classifier on frozen features, run:

```sh
python main_lincls.py \
    -a resnet50 \
    --lr 30.0 \
    --batch-size 256 \
    --pretrained [PATH_TO_CHECKPOINT] \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    [PATH_TO_DATASET]
```

## Reference Validation Accuracy

### ImageNet-100

<table>
   <tr>
      <th rowspan="2"></th>
      <th rowspan="2">Batch Size</th>
      <th rowspan="2">Initial LR</th>
      <th colspan="5">Loss Formula</th>
   </tr>
   <tr>
      <th>
        <img src="https://latex.codecogs.com/svg.latex?\color{White}\mathcal{L}_\mathsf{contrastive}(\tau\mkern1.5mu{=}\mkern1.5mu0.07)" />
      </th>
      <th>
        <img src="https://latex.codecogs.com/svg.latex?\color{White}\mathcal{L}_\mathsf{contrastive}(\tau\mkern1.5mu{=}\mkern1.5mu0.2)" />
      </th>
      <th>
        <img src="https://latex.codecogs.com/svg.latex?\color{White}\shortstack{$2\cdot\mathcal{L}_\mathsf{align}(\alpha\mkern1.5mu{=}\mkern1.5mu2)$\\$+\hspace{3pt}\mathcal{L}_\mathsf{uniform}(t\mkern1.5mu{=}\mkern1.5mu2)$} " />
      </th>
      <th>
        <img src="https://latex.codecogs.com/svg.latex?\color{White}\shortstack{$3\cdot\mathcal{L}_\mathsf{align}(\alpha\mkern1.5mu{=}\mkern1.5mu2)$\\$+\hspace{3pt}\mathcal{L}_\mathsf{uniform}(t\mkern1.5mu{=}\mkern1.5mu3)$} " />
      </th>
      <th>
        <img src="https://latex.codecogs.com/svg.latex?\color{White}\shortstack{$4\cdot\mathcal{L}_\mathsf{align}(\alpha\mkern1.5mu{=}\mkern1.5mu2)$\\$+\hspace{3pt}\mathcal{L}_\mathsf{uniform}(t\mkern1.5mu{=}\mkern1.5mu4)$} " />
      </th>
   </tr>
   <tr>
      <th rowspan="3">Normal</th>
      <th>128</th>
      <th>0.03</th>
      <td>73.12%&nbsp;(MoCo)</td>
      <td>75.54%</td>
      <td>75.44%</td>
      <td><strong>75.62%</strong></td>
      <td>74.52%</td>

   </tr>
   <tr>
      <th>256</th>
      <th>0.03</th>
      <td>68.18%&nbsp;(MoCo)</td>
      <td>69.3%</td>
      <td>68.28%</td>
      <td>69.66%</td>
      <td>69.46%</td>
   </tr>
   <tr>
      <th>256</th>
      <th>0.06</th>
      <td>71.08%&nbsp;(MoCo)</td>
      <td>73.52%</td>
      <td>73.34%</td>
      <td>73.36%</td>
      <td>73.18%</td>
   </tr>
   <tr>
      <th rowspan="3"><div>Strong&nbsp;Aug.</div>+<div>MLP&nbsp;Head</div>+<div>Cosine&nbsp;LR</div></th>
      <th>128</th>
      <th>0.03</th>
      <td>73.92%</td>
      <td>77.54%&nbsp;(MoCo&nbsp;v2)</td>
      <td>77.4%</td>
      <td><strong>77.66%</strong></td>
      <td>76.7%</td>
   </tr>
   <tr>
      <th>256</th>
      <th>0.03</th>
      <td>69.64%</td>
      <td>67.52%&nbsp;(MoCo&nbsp;v2)</td>
      <td>66.92%</td>
      <td>67.44%</td>
      <td>71.42%</td>
   </tr>
   <tr>
      <th>256</th>
      <th>0.06</th>
      <td>73.36%</td>
      <td>76.32%&nbsp;(MoCo&nbsp;v2)</td>
      <td>75.5%</td>
      <td>75.74%</td>
      <td>73.84%</td>
   </tr>
</table>

### ImageNet

<table>
   <tr>
      <th rowspan="2"></th>
      <th rowspan="2">Batch Size</th>
      <th rowspan="2">Initial LR</th>
      <th colspan="5">Loss Formula</th>
   </tr>
   <tr>
      <th>
        <img src="https://latex.codecogs.com/svg.latex?\color{White}\mathcal{L}_\mathsf{contrastive}(\tau\mkern1.5mu{=}\mkern1.5mu0.2)" />
      </th>
      <th>
        <img src="https://latex.codecogs.com/svg.latex?\color{White}3\cdot\mathcal{L}_\mathsf{align}(\alpha\mkern1.5mu{=}\mkern1.5mu2)+\hspace{3pt}\mathcal{L}_\mathsf{uniform}(t\mkern1.5mu{=}\mkern1.5mu3) " />
      </th>
   </tr>
   <tr>
      <th><div>Strong&nbsp;Aug.</div>+<div>MLP&nbsp;Head</div>+<div>Cosine&nbsp;LR</div></th>
      <th>256</th>
      <th>0.03</th>
      <td>67.5%±0.1% (MoCo&nbsp;v2,&nbsp;from&nbsp;<a href="https://github.com/facebookresearch/moco/tree/3631be074a0a14ab85c206631729fe035e54b525#linear-classification">here</a>)</td>
      <td><strong>67.694%</strong></td>
   </tr>
</table>

***Note***: Numbers with <img src="https://latex.codecogs.com/svg.latex?\color{White}\begin{tabular}[b]{@{}c@{}}{\color{black}.}\vspace{-4pt}\\$\mathcal{L}_\mathsf{uniform}$\end{tabular} " align="top" /> are computed without setting `--moco-unif-no-intra-batch`.

## Trained ImageNet Checkpoints

We provide the ResNet50 encoder checkpoint trained on the full ImageNet with <img src="https://latex.codecogs.com/svg.latex?\color{White}\begin{tabular}[b]{@{}c@{}}{\color{Black}.}\vspace{-4pt}\\$3\cdot\mathcal{L}_\mathsf{align}(\alpha\mkern1.5mu{=}\mkern1.5mu2)+\hspace{3pt}\mathcal{L}_\mathsf{uniform}(t\mkern1.5mu{=}\mkern1.5mu3)$\end{tabular}" align="top" />. The encoder is the one achieving 67.694% ImageNet validation top1 accuracy in the table above.

With [PyTorch Hub](https://pytorch.org/docs/stable/hub.html), you may load them without even downloading this repository or the checkpoint:
```py
encoder = torch.hub.load('SsnL/moco_align_uniform:align_uniform', 'imagenet_resnet50_encoder')
```

To load the encoder with the trained linear classifier, use:
```py
encoder = torch.hub.load('SsnL/moco_align_uniform:align_uniform', 'imagenet_resnet50_encoder',
                         with_linear_clf=True)
```

See [here](./hubconf.py#L11-L24) for more details.


Additionally, you may download the saved checkpoints with more information from [here](https://github.com/SsnL/moco_align_uniform/releases/tag/v1.0-checkpoints).


## Acknowledgements and Disclaimer
The code is modified from [the official MoCo repository](https://github.com/facebookresearch/moco).

The ImageNet-100 results included in our paper were **not** computed using this code, since the official MoCo repository was not released at the time of our analysis. Instead, we used a modified version of [the official Contrastive Multiview Coding (CMC) repository](https://github.com/HobbitLong/CMC/), which contains an unofficial implementation of MoCo. There are subtle differences in batch size, learning rate scheduling, queue initialization, etc. In our experience, code provided under this directory can achieve accuracies comparable to the numbers reported in our paper. We encourage readers looking for the exact detailed differences refer to [the appendix of our paper](https://arxiv.org/pdf/2005.10242.pdf#page=22) and [the CMC repository](https://github.com/HobbitLong/CMC/).

We thank authors of [the MoCo repository](https://github.com/facebookresearch/moco) and [the CMC  repository](https://github.com/HobbitLong/CMC/) for kindly open-sourcing their codebases and promoting open research.


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
