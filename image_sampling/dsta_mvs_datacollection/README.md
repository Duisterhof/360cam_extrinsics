# Data collection and pre-processing pipelines for working with fisheye images

## Set up

This project utilizes Git submoudle, so please be careful when you want to introduce any modifications.

For first-time set up:

```bash
git clone git@github.com:castacks/dsta_mvs_datacollection.git
cd dsta_mvs_datacollection
git submodule upate --init --recursive
```

## Dependency

This project makes extensive use of NVIDIA GPU and depends on [PyTorch](https://pytorch.org/) and [CuPy](https://docs.cupy.dev/en/stable/install.html). Please go to the respective web site to install them.

Other dependencies:

```bash
pip3 install numba pyquaternion networkx
```

Note that `numba` might require a special version of `NumPy`.

## Usage

Please refer to [ORD Flexible Data Collection V2 Documentation][data_collection_doc] for detailed usage of the pipelines.

[data_collection_doc]: https://docs.google.com/document/d/1KjhCUnSugCfQJa_XYzeB0sTaklofoqA1yesa0GQoCss/edit?usp=sharing

