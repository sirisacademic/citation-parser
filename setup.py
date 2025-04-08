from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements = []
req_path = Path("requirements.txt")
if req_path.exists():
    requirements = req_path.read_text(encoding="utf-8").splitlines()

# Read long description from README if available
readme_path = Path("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="references-tractor",
    version="1.0.0",
    author="Nicolau Duran-Silva, Pablo Accuosto, Ruggero Cortini",
    author_email="nicolau.duransilva@sirisacademic.com, pablo.accuosto@sirisacademic.com, ruggero.cortini@sirisacademic.com",
    description="Tools for processing raw citations and linking them to scholarly knowledge graphs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sirisacademic/references-tractor",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
