# MCPNet

MCPNet is a gene regulatory network (GRN) reconstruction tool that identify long range indirect interactions based on a novel metric called MCP Score.  MCP score uses maximum-capacity-path, a graph theoretical measure, to quantify the relative strengths of direct and indirect gene-gene interactions.  MCPNet is implemented in C++ and is parallelized for multi-core and MPI multi-node environments.  It is designed to reconstruct networks in unsupervised and semi-supervised manners.

## Installation
At the moment the software is software is released as source code, so it is necessary to compile on your own system.

### prerequisites
Below are the prerequisites.  The example code are for *ubuntu* and *debian* distributions

- A modern c++ compiler with c++11 support. Supports Gnu g++, clang++ and Intel icpc.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sudo apt install build-essential`

- MPI, for example openmpi or mvapich2

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sudo apt install openmpi`

- cmake, and optionally cmake GUI

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sudo apt install cmake cmake-curses-gui`

- HDF5

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sudo apt install hdf5-helpers hdf5-tools libhdf5-dev`


### Getting MCPNet

MCPNet can be downloaded directly from 

https://github.com/AluruLab/MCPNet/releases/tag/paper2022

or the latest version can be accessed via git

`git clone https://github.com/AluruLab/MCPNet.git`

### Building MCPNet

To build MCPNet, run the following from a linux command prompt to set up the build directory and configure the build.  Replace the curly braced directory with your actual directory.   

`mkdir {MCPNet_build_dir}`
`cd {MCPNet_build_dir}`
`cmake {MCPNet_source_dir}`

CMake allows customizing the compile options, including turning on or off support for OpenMP, MPI, HDF5, and SIMD instructions.  We recommend that these be left as default, ON.   The easiest way to change these settings is to use the CMake gui, ccmake.

`ccmake {MCPNet_source_dir}`

Alternatively, command line parameters may be used

`cmake {MCPNet_source_dir} -DUSE_MPI=OFF -DUSE_SIMD=OFF`

Once configured, then run

`make`

You can choose to install into the system bin directory with `make install` but it is not necessary.

### Binaries

The compilation will generate a set of binaries in the `{MCPNet_build_dir}/bin` directory.  The current release includes the following binaries

### Full pipeline
- `mcpnet`:  this is the primary executable that runs the MCPNet pipeline, including MI computation, MCP score calculation, and AUPR and AUROC metrics if groundtruth is supplied.

### Pipeline Components
- `pearson`:  computes the pearson correlation for a given gene expression profile input

- `mi`:  computes the mutual information for a given gene expression profile input

- `mcp`: computes the MCP scores, either for fixed length paths (2, 3, 4) from MI matrix, or from two previously computed maximum path capacity matrices.

- `dpi`: filters the MI matrix based on Data Processing Inequality as implemented in ARACNe and TINGe

- `transform`: performs Stouffer and CLR transforms on input matrix

- `au_pr_roc`: computes the AUC for PR and ROC curves based on a ground truth matrix.


### Utilities

- `convert`: utility to convert between file formats.  Supported formats include HDF5, EXP, and CSV.

- `combine`: perform element-wise basic math operations on two or more matrices of the same dimension.

- `diagonal`: sets the diagonal elements of a matrix to the user specified value

- `select`: select a subset of columns and rows from a matrix.

- `threshold`: filters a matrix element-wise based on user specified thresholds.


## Usage

Each binary has its own help commandline parameter that lists all available parameters. For example

`mcpnet -h`

or 

`mcpnet --help`

Below are common parameters:

- `-i, --input`:  input matrix file

- `-o, --output`:  output matrix file

- `-t, --thread`:  number of threads to use, per compute node

- `-m, --method`:  algorithm to use. This is specific for each binary, if present.  Please use the `-h` switch to see the exact algorithms supported.


## Test data and Tests

The software is distributed with some example datasets in HDF5 format. Note that input and output format are specified via file extension, `exp`, `csv`, or `h5`.  Below are some example tests.

### Convert Files

`{MCPNet_build_dir}/bin/convert -i {MCPNet_source_dir}/data/r10c10.exp -o r10c10.csv`

`{MCPNet_build_dir}/bin/convert -i {MCPNet_source_dir}/data/r10c10.exp -o r10c10.h5`

### Compute MI and Pearson

`{MCPNet_build_dir}/bin/mi -i {MCPNet_source_dir}/data/r10c10.h5 -o r10c10.mi.h5 -m 1`

`{MCPNet_build_dir}/bin/pearson -i {MCPNet_source_dir}/data/r10c10.h5 -o r10c10.pearson.csv`

`{MCPNet_build_dir}/bin/mi -i {MCPNet_source_dir}/data/gnw2000.h5 -o gnw2000.mi.h5 -t 4 -m 2`

### Compute DPI or MCP score

`{MCPNet_build_dir}/bin/mcp -i gnw2000.mi.h5 -o gnw2000.mcp4.h5 -m 3 -t 4`

### Transform MI output

`{MCPNet_build_dir}/bin/transform -i r10c10.mi.h5 -o r10c10.mi.stouffer.h5 -m 2 -t 4`

### Compute AUPR

`{MCPNet_build_dir}/bin/au_pr_roc -i gnw2000.mcp4.h5 -x {MCPNet_source_dir}/data/gnw2000_truenet.csv`

### Compute MCPNet and their AUPRs

The mcpnet executable can accept multiple method parameters and will produce output for each one.  The `-o` or `--output` parameter therefore takes a prefix rather than a filename.

`{MCPNet_build_dir}/bin/mcpnet -i {MCPNet_source_dir}/data/gnw2000.h5 -o gnw2000 -m 1 2 3 -t 4 --mi-method 1 -x {MCPNet_source_dir}/data/gnw2000_truenet.csv`

