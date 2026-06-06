from __future__ import annotations

from pathlib import Path
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).resolve().parent
requirements_text = ROOT_DIR.joinpath("requirements.txt").read_text(encoding="utf-16" if ROOT_DIR.joinpath("requirements.txt").read_bytes()[:2] == b"\xff\xfe" else "utf-8")
install_requires = [
    line.strip()
    for line in requirements_text.splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="customer_feedback_analysis",
    version="0.1.0",
    description="Customer review sentiment analysis and insights pipeline.",
    author="",
    author_email="",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
