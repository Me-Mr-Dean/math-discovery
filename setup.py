"""
Setup configuration for Mathematical Pattern Discovery Engine
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README and requirements
here = Path(__file__).parent
readme_path = here / "README.md"
requirements_path = here / "requirements.txt"

# Read long description
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "AI-powered mathematical pattern discovery engine"

# Read requirements
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]
else:
    # Fallback requirements
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0.0",
        "openpyxl>=3.0.0",
    ]

setup(
    name="math-discovery",
    version="0.1.0",
    author="Mathematical Research Team",
    author_email="research@math-discovery.org",
    description="AI-powered mathematical pattern discovery engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/math-discovery",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "bokeh>=2.0.0",
        ],
        "full": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "bokeh>=2.0.0",
            "jupyter>=1.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "math-discover=src.cli.discover:main",
            "math-generate=src.cli.generate:main",
            "math-analyze=src.cli.analyze:main",
            "math-discover-interactive=scripts.interactive_discovery:main",
            "math-discover-patterns=scripts.discover_patterns:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
