from setuptools import find_packages, setup

setup(
    name="finetune-clip",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tensorboard>=2.12.0",
        "setuptools>=65.5.1",
        "clip @ git+https://github.com/openai/CLIP.git",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    python_requires=">=3.8",
) 