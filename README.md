![banner](./images/banner.png)

# `rv` - A simple web-based robot viewer powered by Viser 

This repository contains `rv`, a simple web-based robot viewer powered by [Viser](https://viser.studio/main/) <img alt="viser logo" src="https://viser.studio/main/_static/logo.svg" width="15" height="auto" />. 


## Features
- Visualize robot models in 3D (currently only supports [URDF](https://wiki.ros.org/urdf) format).
- Interact with the robot via joint and Cartesian controls (powered by [pink](https://github.com/stephane-caron/pink)).
- Access 100+ robot models from [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py).

## Getting Started

### Choose your package manager
`rv` can be easily run using either [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pixi](https://pixi.prefix.dev/latest/installation/). Choose the one you prefer. 

>[!Note] 
>For Windows users, please use pixi as pink cannot be installed on Windows via uv at the moment (Refs: [stephane-caron/pink#138](https://github.com/stephane-caron/pink/issues/138), [stack-of-tasks/pinocchio#2486](https://github.com/stack-of-tasks/pinocchio/issues/2486)).

`main` now supports both package managers. uv and pixi are kept in separate manifests (`pyproject.toml` and `pixi.toml`) to avoid cross-tool dependency conflicts.

### Getting started with uv
```shell
# clone this repository
git clone https://github.com/zixingjiang/robot-viewer.git

# run the viewer
cd robot-viewer
uv run rv
```

#### Installation (optional)

If you’d like to run `rv` from anywhere in your terminal without the uv run prefix, you can install it to your PATH by running the following command in the project directory:

```shell
uv tool install -e .
```

### Getting started with pixi
```shell
# clone this repository
git clone https://github.com/zixingjiang/robot-viewer.git

# run the viewer
cd robot-viewer
pixi run rv
```

### CLI
Please run one of the following to see the available CLI options:
- `uv run rv --help`
- `pixi run rv -- --help`

## Usage
### View local URDF file
Note: Your local file stays on your machine. It won't be uploaded to Internet.

![](./images/open_urdf.gif)

### View robot from robot_descriptions
Note: Internet connection is required to fetch the robot models from thier respective repositories.

![](./images/open_rd.gif)

### Visibility control
![](./images/visibility_control.gif)

### Joint control
![](./images/joint_control.gif)

### Cartesian control
Note: Turn on Cartesian control will disable joint control.

![](./images/cartesian_control.gif)

## License
This repository is publically available under the [MIT License](LICENSE). 
