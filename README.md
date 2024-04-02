# Weight-Shift Regularized Sharpness-Aware Minimization for Gradual Domain Adaptation

**Authors:** Samuel Schapiro, Han Zhao

This is the official implementation for "Weight-Shift Regularized Sharpness-Aware Minimization for Gradual Domain Adaptation". It is largely based on the code from the paper ["Gradual Domain Adaptation: Theory and Algorithms,"](https://arxiv.org/abs/2310.13852) and the ICML 2022 paper ["Understanding gradual domain adaptation: Improved analysis, optimal path and beyond"](https://arxiv.org/abs/2204.08200).

## Algorithm

![Main Result](algo.jpg "Weight Shift Regularized Gradual Self Training (GST) with SAM")


## Results

TO-DO: Insert table/figure with results


# Install the repo
```
git clone https://github.com/samjschapiro/SAM-GDA.git
cd SAM-GDA
pip install -r requirements.txt
```
# Prepare Data

The covertype dataset can be downloaded from: https://archive.ics.uci.edu/dataset/31/covertype. 

The portraits dataset can be downloaded from [here](https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0). We follow the same data preprocessing procedure from https://github.com/p-lambda/gradual_domain_adaptation. Namely after downloading, extract the tar file, and copy the "M" and "F" folders inside a folder called dataset_32x32 inside the current folder. Then run "python create_dataset.py".

# Run Experiment
To run experiments, follow the following syntax.
```
python experiments.py --dataset color_mnist --intermediate-domains 1  --opt-name sam
```
Arguments:
- `dataset` can be selected from `[mnist, portraits, covtype, color_mnist]`
- `optname` can be selected from `[sgd, adam, sam, ssam, asam, fsam, esam]`
- `base-opt` can be selected from `[sgd, adam]`
- `intermediate-domains` is the number of intermediate domains $T$
- `rotation-angle` can be any integer in `[0, 359]`
- `source-epochs` is the number of epochs to use for training on the source domain $t = 0$
- `intermediate-epochs` is the number of epochs to use for training on intermediate domains $t = 1, 2, \dots, T$
- `batch-size` is the batch size for training
- `lr` is the learning rate
- `num-workers` is the number of workers for parallelization



# Citation

This repository is largely based on work done by Haoxiang Wang, Yifei He, Bo Li, and Han Zhao:

```
@misc{he2023gradual,
      title={Gradual Domain Adaptation: Theory and Algorithms}, 
      author={Yifei He and Haoxiang Wang and Bo Li and Han Zhao},
      year={2023},
      eprint={2310.13852},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@inproceedings{wang2022understanding,
  title={Understanding gradual domain adaptation: Improved analysis, optimal path and beyond},
  author={Wang, Haoxiang and Li, Bo and Zhao, Han},
  booktitle={International Conference on Machine Learning},
  pages={22784--22801},
  year={2022},
  organization={PMLR}
}
```
