from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

name = "lm_dataloader"
setup(
    name=name,
    packages=find_packages(),
    version="0.0.2",
    license="MIT",
    description="Dataloader tools for language modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/sdtblck/{name}",
    author="Sid Black",
    author_email="sdtblck@gmail.com",
    install_requires=[
        "torch",
        "numpy",
        "lm_dataformat==0.0.20",
        "ftfy",
        "tqdm",
        "requests",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={"console_scripts": ["lm-dataloader=lm_dataloader.cmd_line:main"]},
)