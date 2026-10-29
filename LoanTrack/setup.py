from setuptools import setup, find_packages

setup(
    name="loantrack",
    version="1.0.0",
    description="LoanTrack — End-to-end Loan Eligibility Prediction System with ML & Web UI",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.23",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "joblib>=1.2",
        "scikit-learn>=1.2",
        "xgboost>=1.7",
        "fastapi>=0.100",
        "uvicorn[standard]>=0.23",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "loan-ml=cli:main",
        ],
    },
)
