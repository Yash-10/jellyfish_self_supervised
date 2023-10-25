# Jellyfish Galaxies: a self-supervised approach for assisting visual identification

[![Notebook example](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I7lIh-vaYhlifvULbzfU3enas008jxre?usp=sharing)

**Motivation**\
Jellyfish galaxies are galaxies undergoing transformation and are characterized by extended tails of gas that have been removed from the galaxy due to a phenomenon called ram pressure stripping. They are rare to find and possess peculiar morphologies. The traditional approach of detecting jellyfish galaxies is visual inspection. However, visual analysis is subjective and is affected by human biases. It is also infeasible to manually vet through large astronomical datasets to detect these rare galaxies. This work aims to assist visual classification using self-supervised learning to improve the quality of the visual classification. Such an approach alleviates human biases and is scalable to large datasets.

## Data

The data (raw images, processed images, and trained models) can be found [here](https://drive.google.com/drive/folders/1HTWDpad8P7trQN_od8FFc6qIdNJ_AfqQ?usp=sharing).

## Repository description
The Colab notebook provides code for how to set up data for training the self-supervised model, presents an example demonstration of the query by example, how the uncertain visual classification of jellyfish galaxies can be improved using self-supervised learning, Grad-CAM analysis, and trail vector analysis of the galaxies. Some miscellaneous code is also present in the notebook.

- `catalogs_analysis` contains catalogs and related codes for generating the plots of the astrophysical analysis.
- `data_utils` contains helper scripts for handling data for the deep learning application.
- `self_supervised` contains code for creating the self-supervised (here, SimCLR) model, evaluation metrics, and other helper functions.
- `old_temp_store` contains codes not used in our analysis but were developed at some point during the work. This folder is kept only for storage purposes.
- `pretrain.py` is the script used to pre-train the self-supervised model.
- `baseline_resnet_supervised.py` contains code for training the supervised CNN model.
- `cross_validate.py` contains code for the K-fold cross-validation for the self-supervised approach, and `cross_validate_linear_eval.py` contains code for the K-fold cross-validation of the linear evaluation protocol (i.e., a linear classifier trained on the self-supervised representations).

**Note**: The procedure for estimating the star formation rates and associated codes is present in the following GitHub repository: https://github.com/amanda-lopes/Halpha-SPLUS-Jelly

## References

The `SimCLR` code in this repository is inspired by the SimCLR tutorial in the UvA DL Notebooks: [Tutorial 17: Self-Supervised Contrastive Learning with SimCLR](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)

## License
[MIT](https://github.com/Yash-10/jellyfish_self_supervised/blob/main/LICENSE)
