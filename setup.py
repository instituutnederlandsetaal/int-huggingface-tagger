# setup.py
from setuptools import setup, find_packages

setup(
    name="int-huggingface-tagger",  # <== Your package name on PyPI (if you ever upload it there)
    version="0.1.0",
    packages=find_packages(),  # automatically finds packages with __init__.py
    description="A Python package for lemmatization and POS tagging of historical Dutch using the transformers library",
    author="Jesse de Does",
    author_email="jesse.dedoes@ivdnt.org",
    url="https://github.com/INL/int-huggingface-tagger",  # your repo link
    # If you have a license file, specify it here:
    license="Undefined-as-yet",
    # List your dependencies here or read them from requirements.txt
    # (But "hardcoding" them is often simpler for library use.)
    install_requires=[
       "transformers",
       "torch==1.11.0+cpu ",
       "torchvision==0.10.0+cpu ",
       "torchaudio==0.9.0 ",
       "transformers[torch]",
       "matplotlib",
       "sklearn",
       "adjustText",
       "wheel",
       "seaborn",
       "sentencepiece",
       "fugashi[unidic]",
       "ipadic",
       "jsons",
       "bottle",
       "nlpaug",
       "datasets",
       "lxml",
       "nltk",
       "pandas",
       "psycopg2-binary",
       "tabulate",
       "evaluate",
       "gitpython"

        # ... etc ...
    ],
    # Optional: if you want to install scripts/ as console commands:
     entry_points={
         "console_scripts": [
            "hugtag-txt=scripts.example_usage:main",
            "hugtag-tei=scripts.example_usage_TEI:main",
            "hugtag-tsv=scripts.example_usage_TSV:main",
         ],
     },
    python_requires=">=3.10",  # or whichever versions you want to support
)

