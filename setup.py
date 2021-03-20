from setuptools import setup

setup(
    name="alfs_char",
    version="0.1.0",
    install_requires=[
        "tqdm",
        "opencv-python",
        "efficientnet_pytorch",
        "torchvision",
        "torch",
        "toolz",
        "vnet"
    ],
    extras_require={
        "develop": [
            "pytest",
            "black",
            "pytest-cov",
            "pytest-benchmark",
            "mypy",
            "kaggle",
            "pandas",
            "cytoolz",
            "torch-optimizer",
            "albumentations",
            "scikit-learn",
        ]
    },
)
