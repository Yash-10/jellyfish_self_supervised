# Jellyfish Self-supervised learning

[![Notebook example](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I7lIh-vaYhlifvULbzfU3enas008jxre?usp=sharing)

## Data

The data (raw images, processed images, and trained models) can be found [here](https://drive.google.com/drive/folders/1HTWDpad8P7trQN_od8FFc6qIdNJ_AfqQ?usp=sharing).

## Repository description
The Colab notebook provides code for how to setup data for training the self-supervised model, presents an example demonstration of query by example, how the uncertain visual classification of jellyfish galaxies can be improved using self-supervised learning, Grad-CAM analysis, and tral-vector analysis of the galaxies. Some miscellaneous code is also present in the notebook.

- `catalogs_analysis` contains catalogs, and related codes for generating the plots of the astrophysical analysis.
- `data_utils` contains helper scripts for handling data for the deep learning application.
- `self_supervised` contains code for creating the self-supervised (here, SimCLR) model, evaluation metrics, and other helper functions.
- `old_temp_store` contains codes not used in our analysis but were developed at some point in the past. This folder is kept only for storage purposes.
- `pretrain.py` is the script used to pre-train the self-supervised model.
- `baseline_resnet_supervised.py` contains code for training the supervised CNN model.
- `cross_validate.py` contains code for the $K$-fold cross validation for the self-supervised approach, and `cross_validate_linear_eval.py` contains code for the $K$-fold cross validation of the linear evaluation protocol (i.e., a linear classifier trained on the self-supervised representations).

## References

The `SimCLR` code in this repository is inspired from the SimCLR tutorial in the UvA DL Notebooks: [Tutorial 17: Self-Supervised Contrastive Learning with SimCLR](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)

## License
[MIT](https://github.com/Yash-10/jellyfish_self_supervised/blob/main/LICENSE)
