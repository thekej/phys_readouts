from setuptools import setup, find_packages, Extension
import os

# Function to find all C++ files in the entire project directory
def find_cpp_files():
    cpp_files = []
    for root, _, files in os.walk('.'):  # Start from the current directory
        for file in files:
            if file.endswith(".cpp"):
                cpp_files.append(os.path.join(root, file))
    return cpp_files

# Find all C++ files in the entire project
cpp_files = find_cpp_files()

# Define the extension module
phys_readouts_extension = Extension(
    name='phys_readouts_extension',  # Name of the extension module
    sources=cpp_files,
    include_dirs=[],  # Add any include directories if needed
    extra_compile_args=['-std=c++11'],  # Add any compiler flags if needed
)

# Setup configuration
setup(
    name='phys_readouts',
    version='0.1',
    packages=find_packages(),  # Automatically finds all packages in the directory
    ext_modules=[phys_readouts_extension],  # Include the extension module
)

