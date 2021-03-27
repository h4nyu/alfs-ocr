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
        "albumentations",
        "vnet",
        "fastapi",
        "uvicorn"
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
            "torch-optimizer",
            "scikit-learn",
        ]
    },
)
