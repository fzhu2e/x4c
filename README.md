# x4c
x4c (xarray4cesm) is an Xarray plugin that aims to support efficient and intuitive CESM output analysis and visualization:
- Analysis features: regrid, various of mean calculation, annualization/seasonalization, etc.
- Visualization features: including timeseries, horizontal and vertical spatial plots, etc.

## Installation

```
conda install -n x4c-env python=3.11
conda activate x4c-env
conda install jupyter notebook xesmf cartopy
pip install x4c
```