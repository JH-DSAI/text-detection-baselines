# text-detection-baselines

[![CI](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/ci.yml/badge.svg)](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/text-detection-baselines/badge/?version=latest)](https://text-detection-baselines.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/JH-DSAI/text-detection-baselines/graph/badge.svg?token=meQRW4r7mP)](https://codecov.io/gh/JH-DSAI/text-detection-baselines)
[![Security](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/security.yml/badge.svg)](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/security.yml)

Benchmarking suite for machine text detection.

## Quickstart (pixi)

1. Install pixi from <https://pixi.sh/latest/>.
1. Clone this repository.
1. Install environments and lockfile dependencies:

```bash
pixi install -a
```

1. Run tests:

```bash
pixi run -e dev test
```

## Common commands

```bash
# lint/format/security
pixi run -e dev check-style
pixi run -e dev check-security
pixi run -e dev format

# tests + coverage outputs
pixi run -e dev test

# docs
pixi run -e docs build-docs

# distribution artifacts (wheel + sdist)
pixi run -e dist build-dist

# run app
pixi run start
```

## Docker

Build:

```bash
docker build -t text-detection-baselines .
```

Run:

```bash
docker run --rm -p 8000:8000 text-detection-baselines
```

## Git hook (optional)

Install the pre-push hook to run style checks before pushing:

```bash
cp ./githooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

## Notes

- CI, security, docs, and distribution workflows use pixi tasks.
- Read the Docs installs documentation dependencies from project extras.
