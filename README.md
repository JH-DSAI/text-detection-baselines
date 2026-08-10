# text-detection-baselines

[![CI](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/ci.yml/badge.svg)](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/text-detection-baselines/badge/?version=latest)](https://text-detection-baselines.readthedocs.io/en/latest/?badge=latest)
[![Security](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/security.yml/badge.svg)](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/security.yml)
<!-- [![DOI](https://zenodo.org/badge/DOI/REPLACE/ME.svg)](https://doi.org/REPLACE/ME) -->

Benchmarking suite for machine text detection.

To do:

* Uncomment and update the DOI above in this README.
* Import package into [readthedocs](https://readthedocs.org/).
* Update [.zenodo.json](.zenodo.json). For more details see [zenodo.json docs](https://developers.zenodo.org/#representation) and [zenodo docs on contributors vs creators](https://help.zenodo.org/docs/deposit/describe-records/contributors/).
* Update quickstart guide below.

## Quickstart (pixi)

1. Install pixi from <https://pixi.sh/latest/>.
1. Clone this repository.
1. Run evaluation on default datasets and models (automatically installs dependencies in a virtual environment):

```bash
pixi run main
```

## Common commands

```bash
# lint/format
pixi run -e dev lint
pixi run -e dev format

# tests + coverage outputs
pixi run -e dev test

# docs
pixi run -e docs build-docs

# distribution artifacts (wheel + sdist)
pixi run -e dist build-dist
```

## Docker

Build:

```bash
docker build -t text-detection-baselines .
```

Run:

```bash
docker run --rm text-detection-baselines
```

## Git hook (optional)

Install the pre-push hook to run style checks before pushing:

```bash
cp ./githooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

## Notes

* CI, security, docs, and distribution workflows use pixi tasks.
* Read the Docs installs documentation dependencies from project extras.
