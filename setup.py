from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = []

setup(
    name="references-tractor",
    version="1.0.0",
    author="Nicolau Duran-Silva, Pablo Accuosto, Ruggero Cortini",
    author_email="nicolau.duransilva@sirisacademic.com, pablo.accuosto@sirisacademic.com, ruggero.cortini@sirisacademic.com",
    description="ReferencesTractor provides tools for processing raw citation and linking to objects in scholarly knowledge graphs.",
    url="https://github.com/sirisacademic/citation-parser",
    packages=find_packages(),  # Automatically find all packages
    python_requires=">=3.9",
    install_requires=requirements,  # Load dependencies from requirements.txt
    include_package_data=True,  # Include additional files specified in MANIFEST.in
)

