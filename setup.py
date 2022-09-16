#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setuptools.setup(
    name="alfahor",
    version="1.0.2",
    author="Teresa Paneque-Carreño",
    author_email='tpaneque@eso.org',
    packages=["alfahor"],
    url="https://github.com/teresapaz/alfahor",
    license="LICENSE.md",
    description=("Vertical structure from masked channel maps."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        "matplotlib",
        "astropy"]
)
