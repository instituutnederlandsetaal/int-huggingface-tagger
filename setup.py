import os
from setuptools import setup, find_packages

# --- Set environment variable (only during installation) ---

os.environ["MY_PACKAGE_ENV"] = "1"
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"]="True"

# --- Read dependencies from requirements.txt ---

def parse_requirements(filename="requirements.txt"):
    with open(filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="int-huggingface-tagger",
    version="0.1.0",
    packages=["lemmatizer", "tagging"],  # explicitly include your two folders
    install_requires=parse_requirements(),
    include_package_data=True,
    description="tagger and lemmatizer for historical Dutch",
    author="Instituut voor de Nderlandse Taal",
    author_email="servicedesk@ivdnt.org",
    url="https://github.com/INL/int-huggingface-tagger"
)

