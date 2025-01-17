
# Ecological connectivity analysis for Switzerland <img src="https://speed2zero.ethz.ch/wp-content/uploads/2023/02/SPEED2ZERO_Logo_trans.png" width="300" align="right">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14676988.svg)](10.5281/zenodo.14676988)


This repository contains the code used to assess the contribution of a location to the ecological connectivity at the Swiss landscape level, within the context of the [SPEED2ZERO](https://speed2zero.ethz.ch/en/) project.

The importance of a pixel for supporting ecological connectivity is evaluated by quantifying how a marginal change in habitat quality and/or permeability affects the overall ecological connectivity of the landscape for a certain taxonomic group. This marginal change is called [quality or permeability *elasticity*](https://en.wikipedia.org/wiki/Elasticity_(economics)#Definition). These elasticities are calculated with the JAX library [`jaxscape`](https://github.com/vboussange/jaxscape).

Maps obtained at the taxonomic group level are aggregated groups to obtain a single **Ecological Connectivity Importance Score**, which identifies areas critical to maintaining connectivity.

![](ecological_connectivity_importance.png)
> Ecological connectivity importance score map. Higher values indicate higher contribution of the pixels for overall ecological connectivity, implying larger loss of connectivity if the pixel's ecological quality or permeability is degraded.

A manuscript detailing the approach will be available soon.

## ⚠️ Disclaimer ⚠️

Elasticity maps and the Ecological Connectivity Importance Score should be considered as work in progress. Prior to the release of version `v1.0.0`, these products are provided solely for research purposes and must not be utilized for conservation planning or decision-making.

## Requirements

#### Hardware
A JAX-compatible GPU is recommended for computational efficiency.

#### Dependencies
To install the dependencies, make sure you have conda (or mamba), go to `python/` and run
```
conda env create --file environment.yml --prefix ./.env
```

#### Input data

The analysis depends on mean suitability maps for each taxonomic group considered, which are available from [this Zenodo archive](https://zenodo.org/records/14676988) and placed under `data/raw`. Simply download the folder and place it under the root folder.

The species maps from which the mean suitability maps have been derived, together with the mean dispersal range used for the calculation of ecological proximity, are stored in each `.nc` file attributes.

Access to the individual species distribution maps used to generate the mean suitability maps for each taxonomic group, along with individual species dispersal range data, is restricted but may be considered upon request.

## File description
- `python/biodiv_layer/group_elasticity_*.py`: Calculate (pemerability/quality) elasticities at the taxonomic group level. 
- `group_summed_elasticities`: Aggregates elasticities to calculate the Ecological connectivity importance score.
- `src/*`: Utility functions.

## Results
Elasticity maps and the Ecological connectivity importance score product are hosted under [this Zenodo archive](https://zenodo.org/records/14676988) and placed under `data/processed/HASH/`. Permeability elasticity maps are provided exclusively for groups where least-cost path distance is used to calculate proximity. For groups where Euclidean distance is used, species movement is assumed to be unaffected by the landscape.

## Roadmap
- [ ] Harmonize `.nc` and `.tiff` file formats.
- [ ] Use `ResistanceDistance` instead of `LCPDistance`, as a more relevant ecological distance.
- [ ] Run connectivity analysis per species, instead of per group