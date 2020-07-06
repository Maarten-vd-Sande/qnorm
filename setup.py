#!/usr/bin/env python
from setuptools import setup
from distutils.util import convert_path
import re

import toml


with open(convert_path('qnorm/__init__.py')) as ver_file:
    match = next(re.finditer('__version__ = "(.*)"', ver_file.read(), re.MULTILINE))
    __version__ = match.group(1)


project = toml.load("pyproject.toml")["project"]

# read the readme as long description
with open("README.md") as f:
    project["long_description"] = f.read()

project["long_description_content_type"] = "text/markdown"
project["version"] = __version__

setup(**project)
