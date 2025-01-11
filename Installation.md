## 📦 Installation

### Prerequisites
- **System**: The code is currently tested only on **Ubuntu 24.04**.  For windows setup, you may refer to [#3](https://github.com/microsoft/TRELLIS/issues/3) (not fully tested).
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA GTX 4090D GPU.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA 12.4.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.12 is required. 

### Installation Steps
1. Install CUDA
    ```sh
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
    ```
    set environment variables
    ```sh
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    ```

2. Create python environment and install pytorch
    设置 conda 源
    新建 conda 配置文件 ${HOME}/.condarc
    添加如下内容。
    ```sh
    channels:
        - defaults
    show_channel_urls: true
    default_channels:
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
    custom_channels:
    conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    ```

    ```sh
    conda create -n trellis python=3.12
    conda activate trellis    
    ```
    设置pip源为清华源
    ```sh
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    ```
    install pytorch
    ```sh
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

3. Install spconv
    You need to ensure ```pip list | grep spconv``` and ```pip list | grep cumm``` show nothing before install editable spconv/cumm.
    install gcc lib
    ```sh
    conda install -c conda-forge libstdcxx-ng
    ```
    if you still encounted "`GLIBCXX_3.4.32' not found" error at runtime, find the libstdc++ lib and copy to python environment lib. for detail, refer to [fix "'GLIBCXX_3.4.32' not found" error at runtime"](https://stackoverflow.com/questions/76974555/glibcxx-3-4-32-not-found-error-at-runtime-gcc-13-2-0)
    ```sh
    cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/miniconda3/envs/trellis/lib/
    ```
   The c++ code will be built automatically when you change c++ code in project.
   
   install cumm:
   ```git clone https://github.com/FindDefinition/cumm```, ```cd ./cumm```, ```pip install -e .```
   install spconv:
   ```git clone https://github.com/traveller59/spconv```,```cd ./spconv```.
   You need to remove ```cumm``` in ```requires``` section in pyproject.toml after install editable ```cumm``` and before install spconv due to pyproject limit (can't find editable installed ```cumm```).
   change `pyproject.toml`, like `requires = ["setuptools>=41.0", "wheel", "pccm>=0.4.16"]`
   install spconv ```pip install -e .```
   in python, ```import spconv``` and wait for build finish.

4. Install trellits and dependencies
    Clone the repos:
    ```sh
    git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
    ```

    Install the dependencies:
    
    **Before running the following command there are somethings to note:**

    - If you have multiple CUDA Toolkit versions installed, `PATH` should be set to the correct version before running the command. For example, if you have CUDA Toolkit 11.8 and 12.2 installed, you should run `export PATH=/usr/local/cuda-11.8/bin:$PATH` before running the command.
    - By default, the code uses the `flash-attn` backend for attention. For GPUs do not support `flash-attn` (e.g., NVIDIA V100), you can remove the `--flash-attn` flag to install `xformers` only and set the `ATTN_BACKEND` environment variable to `xformers` before running the code. See the [Minimal Example](#minimal-example) for more details.
    - The installation may take a while due to the large number of dependencies. Please be patient. If you encounter any issues, you can try to install the dependencies one by one, specifying one flag at a time.
    - If you encounter any issues during the installation, feel free to open an issue or contact us.
    
    
    the dependencies:
    ```sh
    . ./setup.sh --basic --xformers --flash-attn --diffoctreerast --mipgaussian --kaolin --nvdiffrast
    ```
    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
    ```sh
    Usage: setup.sh [OPTIONS]
    Options:
        -h, --help              Display this help message
        --basic                 Install basic dependencies
        --xformers              Install xformers
        --flash-attn            Install flash-attn
        --diffoctreerast        Install diffoctreerast
        --vox2seq               Install vox2seq
        --mipgaussian           Install mip-splatting
        --kaolin                Install kaolin
        --nvdiffrast            Install nvdiffrast
        --demo                  Install all dependencies for demo
    ```

<!-- Pretrained Models -->
## 🤖 Pretrained Models

We provide the following pretrained models:

| Model | Description | #Params | Download |
| --- | --- | --- | --- |
| TRELLIS-image-large | Large image-to-3D model | 1.2B | [Download](https://huggingface.co/JeffreyXiang/TRELLIS-image-large) |
| TRELLIS-text-base | Base text-to-3D model | 342M | Coming Soon |
| TRELLIS-text-large | Large text-to-3D model | 1.1B | Coming Soon |
| TRELLIS-text-xlarge | Extra-large text-to-3D model | 2.0B | Coming Soon |

The models are hosted on Hugging Face. You can directly load the models with their repository names in the code:
```python
TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
```

If you prefer loading the model from local, you can download the model files from the links above and load the model with the folder path (folder structure should be maintained):
```python
TrellisImageTo3DPipeline.from_pretrained("/path/to/TRELLIS-image-large")
```

<!-- Usage -->
## 💡 Usage

### Minimal Example

Here is an [example](example.py) of how to use the pretrained models for 3D asset generation.

```python
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
image = Image.open("assets/example_image/T.png")

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave("sample_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave("sample_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("sample.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply("sample.ply")
```

After running the code, you will get the following files:
- `sample_gs.mp4`: a video showing the 3D Gaussian representation
- `sample_rf.mp4`: a video showing the Radiance Field representation
- `sample_mesh.mp4`: a video showing the mesh representation
- `sample.glb`: a GLB file containing the extracted textured mesh
- `sample.ply`: a PLY file containing the 3D Gaussian representation


### Web Demo

[app.py](app.py) provides a simple web demo for 3D asset generation. Since this demo is based on [Gradio](https://gradio.app/), additional dependencies are required:
```sh
. ./setup.sh --demo
```

After installing the dependencies, you can run the demo with the following command:
```sh
python app.py
```

Then, you can access the demo at the address shown in the terminal.

***The web demo is also available on [Hugging Face Spaces](https://huggingface.co/spaces/JeffreyXiang/TRELLIS)!***


<!-- Dataset -->
## 📚 Dataset

We provide **TRELLIS-500K**, a large-scale dataset containing 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores. Please refer to the [dataset README](DATASET.md) for more details.