from setuptools import setup, find_packages

setup(
    name="ad-blaster",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "Pillow",  # For image processing
        "ollama",
        "opencv-python"
    ],
    author="Ian Marcinkowski",
    author_email="ian@desrt.ca",
    description="A project for blasting ads using images and LLMs",
    url="https://github.com/ianmarcinkowski/ad-blaster",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)