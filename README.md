# Installation Guide

<br>

# Generating synthetic data

## Info

To generate synthetic images you have a lot of methods. You can use Microsoft Image Creator, DALL-E and other image generators but I decided to use ComfyUI. This UI will let you design and execute advanced stable diffusion pipelines using a graph/nodes/flowchart based interface.

## Installing

### Step 1: Download

When you go to this link you can download ComfyUI as a ZIP file. Once it is downloaded, extract the ZIP files.

[GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#installing)

### Step 2: Download a model

In order to start using ComfyUI you need a model. You can download models you like on [civitai](https://civitai.com/models). I used the [DreamShaperXL - Turbo](https://civitai.com/models/112902/dreamshaper-xl).

Once you’ve downloaded a model, put it in this folder:

`ComfyUI_windows_portable\ComfyUI\models\checkpoints`

### Step 3: Start ComfyUI

If you have an Nvidia GPU: Run `run_nvidia_gpu.bat` to run ComfyUI

If you don’t have an Nvidia GPU: Run `run_cpu.bat` to run ComfyUI (very slow)

### (Optional) Step 4: Use a workflowa

For my project I used a workflow that used SDXL Turbo with DreamShaper. This gave me very good and quick results. You can download the workflow with the following link.

[SDXL Turbo - Dreamshaper | ComfyUI Workflow | OpenArt](https://openart.ai/workflows/barrenwardo/sdxl-turbo---dreamshaper/v3MNWyfGSlS8VeZKMp1g)

In order to use this workflow, I recommend you to install the ComfyUI manager. With the ComfyUI manager you can do all sorts of things like installing missing custom nodes.

Open a terminal and change your directory to `ComfyUI_windows_portable\ComfyUI\custom_nodes` and then run the following command.

```python
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

Once it is cloned restart your ComfyUI and you should have a button called `Manager`

Now you can import your workflow file by clicking on the load button and then selecting the json workflow file.

Once it is imported you will see that some custom nodes are missing but you can install them with the manager you just installed. Just click on manager → install missing custom nodes and then select the missing nodes to install. You will have to restart your ComfyUI after.

# Code environment

## Info

For my project I used YOLOv8 to train an object detection model on a custom dataset. I used Roboflow for easy management of my different datasets and models.

## Installation

### Step 1: Creating an environment

For this project I used Anaconda and linked it to visual studio code. I recommend to use it as well. You can download anaconda here: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual) and you can download visual studio code here: [https://code.visualstudio.com/download](https://code.visualstudio.com/download).

When you have installed Anaconda, open Anaconda Prompt and create a new environment with the following command:

```python
conda create --name <env>
```

Once the environment is created, activate it with this command.

```python
conda activate <env>
```

Replace `<env>` with the name of your environment. For example: `conda create --name YOLOv8`.\*\*\*\*

### Step 2: Installing conda packages

Before we install the packages, I recommend you to install [CUDA toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) and [cuDNN](https://developer.nvidia.com/cudnn) if you have a Nvidia GPU. Make sure your cuDNN version is compatible with your CUDA version. You can find the compatible versions here: [https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html).

To install the packages when using a `CPU`, run this command in your anaconda navigator.

```python
conda install -c conda-forge ultralytics
```

To install the packages when using a `GPU`, run this command in your anaconda navigator.

```python
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

### Step 3: Installing additional pip packages

To be able to run all of the code, you will need to install some extra dependencies. Run the following command in your anaconda navigator. Make sure you are still in the right conda environment.

```python
pip install -r requirements.txt
```

### Step 4: Running the project

After you installed all the dependencies, you should be able to run everything. Once you open a notebook file, make sure you select the environment you created to run the code.
