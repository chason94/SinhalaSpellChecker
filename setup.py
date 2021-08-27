from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    'tqdm',
    'torch>=1.6.0',
    'numpy',
    'jsonlines',
    'sentencepiece',
    'sinling'
]

setup(
    name="SinhalaSpellCorrector",
    version="0.1.0",
    author="Charana Sonnadara",
    author_email="csonnadara02@gmail.com",
    description="Sinhala Spell Corrector Based on NeuSpell",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/chason94/SinhalaSpellChecker",
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">3.5",
    install_requires=requirements,
    extras_require={
        "spacy": ["spacy"],
        "elmo": ["allennlp==1.5.0"],
        "noising": ["unidecode"],
        "tokenizer": ["sinling"]
    },
    keywords="transformer networks neuspell neural spelling correction embedding PyTorch NLP deep learning"
)
