#setup.py 
from pathlib import Path
from setuptools import find_namespace_packages, setup

# load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

#setup 
setup(
    name="tagifai",
    version="0.1",
    description="MLOps Project: classify machine learning",
    author="PTSon2207",
    author_email="sonphukl@gmail.com",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
) 
