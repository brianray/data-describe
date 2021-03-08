import os
from typing import Dict

from setuptools import find_packages, setup

NAME = "data_describe"
DESCRIPTION = "A Pythonic EDA Accelerator for Data Science "
URL = "https://data-describe.ai/"
EMAIL = ""
AUTHOR = "https://github.com/data-describe/data-describe/graphs/contributors"
REQUIRES_PYTHON = ">=3.6"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/data-describe/data-describe/issues",
    "Documentation": "https://data-describe.ai/docs/master/",
    "Source Code": "https://github.com/data-describe/data-describe",
}
LONG_DESCRIPTION = """data-describe is a Python toolkit for Exploratory Data Analysis (EDA). It aims to accelerate data exploration and analysis by providing automated and polished analysis widgets."""
version: Dict[str, str] = {}
with open(os.path.join("data_describe", "_version.py")) as fp:
    exec(fp.read(), version)

REQUIRED = [
    "pandas>=1.0",
    "numpy>=1.16",
    "scipy>=1.1",
    "scikit-learn>=0.23",
    "seaborn>=0.11",
    "plotly>=4.0",
    "importlib-metadata;python_version<='3.7'",
]

EXTRAS = {
    "nlp": ["nltk>=3.4", "pyldavis>=2.1.2", "gensim>=3.4.0", "tqdm>=4.49.0"],
    "gcp": ["gcsfs>=0.2.1", "google-cloud-storage>=1.18.0"],
    "pii": ["presidio-analyzer==0.3.8917rc0"],
    "modin": ["modin>=0.7.3", "ray>=0.8.4"],
    "cluster": ["hdbscan>=0.8.17"],
    "time": ["statsmodels>=0.10"],
    "scatter": ["pyscagnostics>=0.1.0a4"],
    "anomaly": ["pmdarima>=1.7"],
}
EXTRAS["all"] = list(set([x for req in EXTRAS.values() for x in req]))


setup(
    name=NAME,
    version=version["__version__"],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls=PROJECT_URLS,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    entry_points={
        "data_describe_viz_backends": [
            "seaborn = data_describe.backends.viz:_seaborn",
            "plotly = data_describe.backends.viz:_plotly",
            "pyLDAvis = data_describe.backends.viz:_pyLDAvis",
        ],
        "data_describe_compute_backends": [
            "pandas = data_describe.backends.compute:_pandas",
            "modin = data_describe.backends.compute:_modin",
        ],
    },
)
