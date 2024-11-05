from setuptools import setup, find_packages

setup(
    name="NEBULANN",
    version="0.2",
    description="Error injection and quantized training lib for NNs",
    author="Alexander Tepe",
    author_email="alexander.tepe@hotmail.de",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "setuptools",
        "tensorflow",
        "numpy",
        "python-dotenv"
    ],
)
