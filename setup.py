#!/usr/bin/env python
from setuptools import setup
from distutils.util import convert_path
import re

import toml


project = toml.load("pyproject.toml")["project"]

# read the readme as long description
with open("README.md") as f:
    project["long_description"] = f.read()

project["long_description_content_type"] = "text/markdown"
with open(convert_path('qnorm/__init__.py')) as ver_file:
    match = next(re.finditer('__version__ = "(.*)"', ver_file.read(), re.MULTILINE))
    project["version"] = match.group(1)
project["data_files"] = [("", ["LICENSE", "pyproject.toml"])]
project["entry_points"] = {
    'console_scripts': ['qnorm=qnorm.cli:main'],
}
setup(**project)
