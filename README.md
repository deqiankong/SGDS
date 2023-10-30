# Sampling with Gradual Distribution Shifting (SGDS)
This is the repository for our paper "Molecule Design by Latent Space Energy-based Modeling and Gradual Distribution Shifting" in UAI 2023. [PDF](https://proceedings.mlr.press/v216/kong23a/kong23a.pdf)

![alt text](https://github.com/deqiankong/SGDS/blob/main/figure/model.png)

In this paper, we studied the following property optimization tasks:
* single-objective p-logP maximization
* single-objective QED maximization
* single-objective ESR1 binding affinity maximization
* single-objective ACAA1 binding affinity maximization
* multi-objective (ESR1, QED, SA) optmization
* multi-objective (ACAA1, QED, SA) optmization

<p align="center">
  <img src="https://github.com/deqiankong/SGDS/blob/main/figure/single.png" width="450", height="200">
  <img src="https://github.com/deqiankong/SGDS/blob/main/figure/multi.png" width="350",, height="200">
</p>

## Enviroment
We follow the previous work [LIMO](https://github.com/Rose-STL-Lab/LIMO) for setting up RDKit, Open Babel and AutoDock-GPU. We extend our gratitude to the authors for their significant contributions.

## Data
We use selfies representations of ZINC250k with corresponding property values. All the property values can be computed either by RDKit or AutoDock-GPU.

## Usage
For model training given certain property (i.e. ESR1),
```
cd single_design_esr1
python main.py
```

For property optimizaton task,
```
python single_design.py or multi_design.py
```

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
