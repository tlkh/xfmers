import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xfmers",
    version="0.0.3",
    author="Timothy Liu",
    author_email="tlkh.xms@gmail.com",
    description="Quickly initialize bespoke Transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tlkh/xfmers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6',
)
