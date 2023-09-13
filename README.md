# TVM SYCL Backend

Add a new backend language——SYCL to TVM, enhancing TVM's compatibility and portability across different types of accelerators.

How to use？Similar to other backends, only need to specify `target='sycl'`.

## TVM-SYCL Install Guideline

### Prerequisites

- `llvm`：> 5.0

- `llvm-sycl`：SYCL compiler。

  - See [dpc++'s installation guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md).

  - Generally, only the following instructions are needed.

    ```shell
    export DPCPP_HOME=~/sycl_workspace
    mkdir $DPCPP_HOME && cd $DPCPP_HOME
    git clone --depth 1 https://github.com/intel/llvm -b sycl
    mkdir $DPCPP_HOME/DPC++
    
    #for nvidia gpu
    python $DPCPP_HOME/llvm/buildbot/configure.py --cuda -o $DPCPP_HOME/DPC++
    #for amd gpu
    python $DPCPP_HOME/llvm/buildbot/configure.py --hip -o $DPCPP_HOME/DPC++
    #for intel gpu
    python $DPCPP_HOME/llvm/buildbot/configure.py -o $DPCPP_HOME/DPC++
    
    python $DPCPP_HOME/llvm/buildbot/compile.py -o $DPCPP_HOME/DPC++
    ```
### Compile
- Get Source from Github

  It is important to clone the submodules along, with `--recursive` option.

  ```shell
  git clone --recursive https://github.com/RELOAD22/tvm tvm
  cd tvm
  ```

- Set configuration options.

  The configuration of TVM can be modified by editing config.cmake. 

  First create a build directory and copy `cmake/config.cmake` to this directory.

  ```shell
  mkdir build && cp cmake/config.cmake build
  ```

  Edit config.cmake:

  ```shell
  cd build && vim config.cmake
  ```

  - Set `set(USE_LLVM)` to `set(USE_LLVM /path/to/your/llvm/bin/llvm-config)`.
  - Enable SYCL related options.
    - Set `set(USE_SYCL)` to the DPC++ path so that `${USE_SYCL}/bin/clang++` points to the clang++ compiler;
    - Set `set(SYCL_GPU)` to the actual GPU type, the optional values are "nvidia", "amd", "intel".
      - for amd gpu，need to specify gpu model. MI50 -> gfx906, MI100 -> gfx908, MI250x -> gfx90a. 
    - `SYCL_TEMP_FOLDER` is a temporary path to store SYCL code and does not need to be modified.

- Build the shared libraries, namely libtvm.so and libtvm_runtime.so.

  ```shell
  cmake ..
  make -j 16
  ```

  After the compilation is successful, two shared library files, libtvm.so and libtvm_runtime.so, appear in the build folder.

- Set the Python path to call the shared library.

  Modify ~/.bashrc and add the following content.

  ```shell
  export TVM_HOME=/path/to/tvm
  export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
  ```

  path/to/tvm is the previously cloned TVM path.

- Install python dependencies

  ```shell
  pip3 install --user numpy decorator attrs
  pip3 install --user tornado psutil xgboost==1.5.0 cloudpickle
  pip3 install --user onnx onnxoptimizer
  ```

  Note that the `--user` flag is not necessary if you’re installing to a managed local environment, like `virtualenv`.

The installation is complete!



## TVM-SYCL Code Example

The following sample code shows that the matrix multiplication example is executed with the CUDA and SYCL backends respectively, and compares whether the results of the two backends are consistent.

```python
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_executor
import tvm.testing
import tvm

# define GEMM
M = 1024
N = 1024
data_shape = (M, N)
dtype = 'float32'
X1 = relay.var("X1", shape=data_shape, dtype=dtype)
X2 = relay.var("X2", shape=data_shape, dtype=dtype)
Y_gemm = relay.nn.dense(X1, X2)
mod = tvm.IRModule.from_expr(Y_gemm)
# initialize input
X1_np = np.random.uniform(size=data_shape).astype(dtype)
X2_np = np.random.uniform(size=data_shape).astype(dtype)

def build(target:str):
    # model build
    tgt = tvm.target.Target(target=target, host="llvm")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=tgt, params=None)
    # print CUDA/SYCL source code
    # print(lib.get_lib().imported_modules[0].get_source()) 
    dev = tvm.device(target, 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("X1", X1_np)
    module.set_input("X2", X1_np)
    module.run()
    tvm_output = module.get_output(0).numpy()
    return tvm_output
    
cuda_output = build(target="cuda")
sycl_output = build(target="sycl")
tvm.testing.assert_allclose(cuda_output, sycl_output, rtol=1e-5, atol=1e-5)
```
only need to specify `target='sycl'`!

## Feedback
If you encounter any issues, you can report them by opening an issue or sending the details to liuyi22s@ict.ac.cn.

This feature was developed by the En Shao (shaoen@ict.ac.cn) team from the Institute of Computing Technology, CAS. Students who are interested in TVM and SYCL are welcome to join us.
