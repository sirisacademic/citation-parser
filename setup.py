from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = []

setup(
    name="citation-parser",
    version="1.0.0",
    author="Nicolau Duran-Silva, Pablo Accuosto, Ruggero Cortini",
    author_email="nicolau.duransilva@sirisacademic.com, pablo.accuosto@sirisacademic.com, ruggero.cortini@sirisacademic.com",
    description="CitationParser provides tools for processing raw citation and linking to objects in scholarly knowledge graphs.",
    url="https://github.com/sirisacademic/citation-parser",
    packages=find_packages(),  # Automatically find packages within citation_parser/
    package_dir={"": "citation-parser"},  # Ensure the module path is correct
    install_requires=requirements,  # Load dependencies from requirements.txt
    python_requires=">=3.9",
    include_package_data=True,  # Include additional files specified in MANIFEST.in
)