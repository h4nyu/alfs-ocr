from setuptools import setup

setup(
    name="alfs_char",
    version="0.1.0",
    install_requires=[
        "tqdm",
        "efficientnet_pytorch",
        "torchvision",
        "torch",
        "toolz",
        "albumentations[imgaug]",
        "torch-optimizer",
        "vision_tools @ git+https://github.com/h4nyu/vision-tools.git@0e5172486a65237a6a6ab9e631e35e5a06355e37#egg=vision_tools&subdirectory=vision_tools",
        "fastapi",
        "uvicorn",
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
            "scikit-learn",
            "tensorboard",
        ]
    },
)
