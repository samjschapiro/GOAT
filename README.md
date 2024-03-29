## Sharpness-Aware Generalization Bounds for Gradual Domain Adaptation

Authors: Samuel Schapiro, Han Zhao

This is the official implementation for studying Sharpness-Aware Minimization in Gradual Domain Adaptation. It is largely based on the code from the paper ["Gradual Domain Adaptation: Theory and Algorithms,"](https://arxiv.org/abs/2310.13852) and the ICML 2022 paper ["Understanding gradual domain adaptation: Improved analysis, optimal path and beyond"](https://arxiv.org/abs/2204.08200).

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
- `intermediate-domains`
- `seed`
- `rotation-angle`
- `source-epochs`
- `intermediate-epochs`
- `batch-size`
- `lr`
- `num-workers`

Here, `dataset` can be selected from `[mnist, portraits, covtype, color_mnist]`. The choice of `opt-name` could be one of 



# Citation

**This repository is largely based on the work from**
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
