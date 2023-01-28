from setuptools import setup, find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()

VERSION = '0.1'
setup(
    name='arithmetic_compressor',
    version=VERSION,
    author='Biereagu Sochima',
    author_email='<sochima.eb@gmail.com>',
    description='An implementation of the Arithmetic Coding algorithm in Python, along with advanced models like PPM (Prediction by Partial Matching), Context Mixing and Simple Adaptive models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kodejuice/arithmetic_compressor',
    packages=find_packages(),
    test_suite='unittest.TestLoader().discover("tests")',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    keywords=['arithmetic', 'coding', 'ppm', 'encoding', 'encoder',
              'prediction', 'context mixing', 'adaptive models'],
    python_requires='>=3.0',
)
