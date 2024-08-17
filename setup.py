from setuptools import setup, find_packages

setup(
    name='phys_readouts',
    version='0.1',
    packages=find_packages(),  # Automatically finds all packages in the directory
    install_requires=[
        # Add any dependencies here
    ],
    include_package_data=True,  # Ensure that non-Python files specified in MANIFEST.in are included
    package_data={
        # If you need to include non-Python files, specify them here
        '': ['*.txt', '*.md'],
        'phys_readouts': ['models/*', 'utils/*'],  # Include all files under models and utils
    },
    entry_points={
        'console_scripts': [
            # Example: 'cli_name = phys_readouts.module:function',
        ],
    },
)
