from setuptools import setup, find_packages

setup(
    name="loan_prediction",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "lightgbm",
        "pyarrow",  # for parquet
        "dalex",    # for model interpretation
        "pytest",
    ],
)