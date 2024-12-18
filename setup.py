import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NaroNet",
    version="0.0.12",  # Incremented version to reflect changes
    author="Daniel Jiménez-Sánchez",
    author_email="danijimnzs@gmail.com",
    description="NaroNet: discovering tumor microenvironment elements from multiplex imaging.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djimenezsanchez/NaroNet",
    project_urls={
        "Bug Tracker": "https://github.com/djimenezsanchez/NaroNet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",  # Updated to Python 3.8
        "Programming Language :: Python :: 3.9",  # Added Python 3.9
        "Programming Language :: Python :: 3.10", # Added Python 3.10
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    license='BSD 3-Clause License',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",  # Updated Python requirement
    install_requires=[
        "numpy>=1.20,<2.0.0",  # Added NumPy version constraint
        "matplotlib>=3.2.1",
        "pandas>=1.1.5",
        "seaborn>=0.11.0",
        "scikit-learn",
        "scikit-image",
        "tifffile>=2020.2.16",
        "pycox>=0.2.0",
        "sklearn-pandas>=2.0.3",
        "torchtuples>=0.2.0",
        "opencv-python>=4.2.0",
        "ray[tune]",
        "xlsxwriter>=1.1.5",
        "imgaug>=0.4.0",
        "xlrd>=1.2.0",
        "tensorboard>=2.0.0",  # Updated to match TensorFlow 2.x
        "argparse>=1.1",
        "hyperopt>=0.2.3",
#        "tensorflow==2.14.0",  # Updated TensorFlow version
        "imbalanced-learn==0.10.1",  # Consolidated imblearn
        "imagecodecs",
        # PyTorch Geometric dependencies are managed separately
    ],
    extras_require={
        "torch_geometric": [
            "torch-scatter==2.0.9",
            "torch-sparse==0.6.13",
            "torch-cluster==1.6.0",
            "torch-spline-conv==1.3.0",  # Updated to a compatible version
            "torch-geometric==2.6.1",  # Updated to a compatible version
        ],
    },
)
