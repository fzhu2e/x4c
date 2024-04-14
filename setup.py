from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='x4c',  # required
    version='2024.4.14b',
    description='xarray4cesm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Feng Zhu, Jiang Zhu',
    author_email='fengzhu@ucar.edu, jiangzhu@ucar.edu',
    url='https://github.com/fzhu2e/x4c',
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
    zip_safe=False,
    keywords='Xarray, CESM, Climate Data Analysis and Visualization',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=[
        'netCDF4',
        'xarray',
        'dask',
        'nc-time-axis',
    ],
)
