conda env create -f environment.yaml
conda activate PANY

# Install Pointnet2
cd vggt/mv_match/model/pointnet2
python -m pip install . --no-build-isolation
cd ../../../..

# Install BOP Toolkit
cd bop_toolkit
pip install -e .
cd ..

