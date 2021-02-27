# MSci-Project-HiggsTauTau

MSci Project with Kristof Galambos

## Requirements:

- conda install pytables

## To do (07 Feb)

- [ ] Tune hyperparameters on all channels, both gen and reco - Kristof
  - [ ] Compare hyperopt performance with sklearn
  - [ ] Coarse scan -> fine scan
  - [ ] Play around with other HPs
    - LRelu, PRelu, Swish
    - Learning rate, momentum
    - Optimiser (?)
  - [ ] Extend paramterspace (Thursday)
- [ ] Integrate additional information - Stanley (Thursday/ Friday)
  - [ ] Neutrino information
    - [ ] Fix imputer KNN and Linear modes
  - [ ] Phi tau
  - [x] Double check if aco angles are redundant
  - [x] PV/IP/SV
  - [x] MET
- [x] Tensorboard
  - [x] Add naming to tensorboard
- [ ] XGBoost comparison - Stanley
- [ ] Smearing of gen level data - Kristof
  - [ ] How to smear? What combinations to smear?
  - [ ] What function to smear with?
- [x] Rename evaluator .png paths
