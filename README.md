# img2pose-visualizer

Just run and evaluate img2pose on your webcam.


## Installation

To install dependencies you should have python 3.12+ installed

Then you can run

```sh
# Create & activate Virtual environment
python -m venv .venv && \
source .venv/bin/activate
```

Then install poetry (package manager)
```sh
pip install poetry
```

After, you can install the dependencies:

```sh
poetry install
```

### Img2pose Installation

For model instalation (in folder `/models`) you should proceed to the [original im2pose repo and recommendations](https://github.com/vitoralbiero/img2pose?tab=readme-ov-file#testing).


## Run

Just run in your terminal 

```sh
python main.py
```

<img src="./assets/img2pose-kevin-kesc23-teste.gif" alt="Kesc23 img2pose test" width="500">

As you can see, the a box with stats including yaw, roll and pitch angles appear whenever a face is detected.

And this should be everything.

## Other info

The dependencies for running img2pose and this visualizer include the following packages (and more):

![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)