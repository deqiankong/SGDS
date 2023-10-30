# Sampling with Gradual Distribution Shifting (SGDS)
This is the repository for our paper "Molecule Design by Latent Space Energy-based Modeling and Gradual Distribution Shifting" in UAI 2023. [PDF](https://proceedings.mlr.press/v216/kong23a/kong23a.pdf)

![alt text](https://github.com/deqiankong/SGDS/blob/main/model.png)

In this paper, we studied the following property optimization tasks:
* single-objective p-logP maximization
* single-objective QED maximization
* single-objective ESR1 maximization
* single-objective ACAA1 maximization
* multi-objective (ESR1, QED, SA) optmization
* multi-objective (ACAA1, QED, SA) optmization

## Enviroment
We follow the previous work [LIMO](https://github.com/Rose-STL-Lab/LIMO) for setting up RDKit, Open Babel and AutoDock-GPU. We extend our gratitude to the authors for their significant contributions.

## Data

## Cite
<pre>
@inproceedings{kong2023molecule,
  title={Molecule Design by Latent Space Energy-Based Modeling and Gradual Distribution Shifting},
  author={Kong, Deqian and Pang, Bo and Han, Tian and Wu, Ying Nian},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={1109--1120},
  year={2023},
  organization={PMLR}
}
</pre>
