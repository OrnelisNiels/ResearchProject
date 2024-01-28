# User Guide

# Generating synthetic data

---

## Introduction

If you have completed all the steps from the installation guide on how to install ComfyUI, then you can follow this user guide on how to generate the images. In this guide I will use a workflow, so make sure you have completed the workflow step in the installation guide

## Dashboard

### Load the workflow

To make it easier just download the following image and just drag it in your ComfyUI dashboard.

![Untitled](User%20Guide/Untitled.png)

If you did that successfully you now should have a dashboard that looks like this.

![Untitled](User%20Guide/Untitled%201.png)

### Generate an image

When you have loaded the workflow, click on the manager button and install missing custom nodes. Restart your ComfyUI when installed. After that you are pretty much good to go. You can toy with the different nodes if you want to, but if you want to generate images you don’t really need to change much.

The most important thing is your positive prompt (Primitive). Your positive prompt is the description of what your image should look like. Then you have your negative prompt inside the Efficient Loader node. This is what the image shouldn’t be. Here you can add things like deformation, unrealistic, and other things you don’t want to see on your image. You can also change the resolution if you like.

To generate an image all you need to do is click the `Queue Prompt` button and your image will start generating. If you want to generate images constantly, just click on `extra options` and then select `auto queue`. If you work with auto queue I recommend you to click on randomize or set your noise_seed to -1 so that you don’t generate the same image over and over again.

Your images are saved here `ComfyUI_windows_portable\ComfyUI\output` .

If you want to understand more of what everything is, check out the node info section.

### Node info

Below you can find the different nodes you have on your dashboard and what they are.

- SDXLResolution
  The SDXLResolution node is an input that goes into the Efficient Loader. This node just defines the width and height the image should be. You can just click on the resolution and select a resolution you like.
- Primitive
  The primitive is a positive text prompt. Here you can type your prompt of what the image should look like. This is also an input that goes into the Efficient Loader.
- Efficient Loader
  - `Checkpoint Loader`: The Efficiency Loader node allows users to select checkpoints from their model checkpoint folder, providing flexibility for model selection.
  - `VAE`: Provides a visual representation of the end result during rendering.
  - `Clip Skip`: Efficiency Nodes incorporate a clip skip feature that allows for skipping unnecessary calculations during rendering, optimizing the rendering process.
  - `Lora Load`: An upcoming feature in Efficiency Nodes, Lora Load, will provide users with the functionality to load models using Lora technology.
  - `Prompts`: You can set your positive and negative prompt of how your image should and shouldn’t look like.
  - You can also set the way you want the prompt to be encoded via the `token_normalization` and `weight_interpretation` widgets.
  - The `batch size` is the number of samples that are processed by the diffusion model at a time.
- Steps
  The number of steps to use during denoising. The more steps the sampler is allowed to make the more accurate the result will be. This is an input for KSampler Advanced.
- CFG
  The classifier free guidance(cfg) scale determines how aggressive the sampler should be in realizing the content of the prompts in the final image. Higher scales force the image to better represent the prompt, but a scale that is set too high will negatively impact the quality of the image. This is an input for KSampler Advanced.
- KSampler Advanced
  - `add_noise`: Wether or not to add noise to the latent before denoising. When enabled the node will inject noise appropriate for the given start step.
  - `seed`: The random seed used in creating the noise`.`
  - `sampler_name`: Which sampler to use, see the [samplers](https://blenderneko.github.io/ComfyUI-docs/Core%20Nodes/Sampling/samplers/) page for more details on the available samplers.
  - `scheduler`: The type of schedule to use, see the [samplers](https://blenderneko.github.io/ComfyUI-docs/Core%20Nodes/Sampling/samplers/) page for more details on the available schedules.
  - `start_at_step`: Determines at which step of the schedule to start the denoising process.
  - `end_at_step`: Determines at which step of the schedule to end denoising. When this settings exceeds `steps` the schedule ends at `steps` instead
  - `return_with_leftover_noise` : When disabled the KSampler Advanced will attempt to completely denoise the latent in the final step. Depending on how many steps in the schedule are skipped by this, the output can be inaccurate and of lower quality.
- Save Image
  Save Image is just what the name says, to save an image. You can change the name of the file with `filename_prefix`.

# Code environment

---

### Introduction

To use the code environment is pretty straightforward. I have filtered all the code that could be useful. The 2 most useful files are`object_detection.ipynb` and `autolabelers.ipynb` .

In `object_detection.ipynb` you can find the structure I used to train multiple object detection models and automatically label synthetic data. To train multiple models you will of course have to re-run the same code but with different datasets/parameters.

In `autolabelers.ipynb` you can find a few other auto-labellers I experimented with and that actually work. If you want to look for other auto-labellers you can find some [here](https://docs.autodistill.com/supported-models/#object-detection).

### object_detection.ipynb

The file is pretty simple. You can run the code of this file if you change the parameters. To train object detection models I used Roboflow for managing my datasets and models easily, whether you do it or not is up to you, but in this guide it’s with Roboflow.

If you have a labelled dataset start from step 2. If you don’t have a labelled dataset yet, start from step 1.

1. Auto-label data

   If you have hundreds or even thousands of images, then an auto-labeller might come in handy. In the file you can find it at 2.1. The code uses a custom function that utilises OWLv2 with Non-max-suppression to reduce overlapping labels. The function has 3 important parameters.

   - `input_folder` : The folder with the images you want to label.
   - `output_folder` : The folder where you want to store your labelled dataset in.
   - `nms_treshold` : This is the threshold to reduce overlapping labels.

   If you don’t know what nms threshold you should take, then run the small application I made to understand that threshold.

   You can run the application by opening anaconda prompt and activating your correct environment. Then you change your directory to the `owlv2_gui` folder and then you run the following command: `python app2.py` and go to localhost:7860 in your browser. Then you should see something like this.

   ![Untitled](User%20Guide/Untitled%202.png)

   Now you just upload an image you want to label. Then you enter a prompt of what you want to label and then you test different threshold values and see which one gives the best results.

2. Import dataset

   First you download a dataset from Roboflow. To do this easily you just create a project on Roboflow and upload your images and labels folder and Roboflow will import it as labelled images. Then you generate a version and then you get code to download your dataset. A low number results in less overlap, but sometimes it ruins your labelling.

3. Train custom model

   The next step is to train your custom model.

   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")

   results = model.train(data="r50s50-25-1/data.yaml", epochs=300, device='0', patience=20, iou=0.2)
   ```

   This is really simple. You just select a pretrained a model you want to use, for example `yolov8n.pt`. This is the smallest model. Here are the other (bigger) models.

   ![Untitled](User%20Guide/Untitled%203.png)

   Next you put the path to your data.yaml file of your dataset and adjust the parameters to what you like. I used the following parameters:

   - `data` : This parameter specifies the path to the YAML file containing the configuration details for the training data.
   - `epochs` : The **`epochs`** parameter defines the number of times the entire dataset is passed forward and backward through the neural network during training.
   - `device` : This parameter specifies the device on which the training will take place. In machine learning, this often refers to the GPU device. 0 is in my case my GPU.
   - `patience` : This represents the number of epochs with no improvement after which training will be stopped. This is called early stopping.
   - `iou`: This stand for Intersection over Union (IoU) , and this is to reduce the overlap between the predicted bounding box and the ground truth bounding box to the area of their union. If you have too many overlapping boxes you can try to lower the value and see if it helps.

   If you want to use more parameters go to [this](https://docs.ultralytics.com/modes/train/) link.

   After the model has been trained, you will be able to find the results in the location `runs/detect/train`

4. Validation & Inference

   For the validation you should define a static testset. This exact testset should be used to validate all your models you train to see which one performs the best. The validations are located a folder like`runs/detect/val` .

   If you want to see how well the model performs on a video or image you can run inference. This is an example of how you can do it in CLI.

   ```python
   !yolo task=detect mode=predict model="runs/detect/full/weights/best.pt" source="D:/ComfyUI/ComfyUI_windows_portable_nvidia_cu121_or_cpu/ComfyUI_windows_portable/ComfyUI/output_forest/ComfyUI_00901_.png" conf=0.25 line_width=1
   ```

   The parameter `model` is just the path to your best model. The `source` is the path to the image or video you want to run inference on. The `conf` is just the confidence threshold, in my case it will put a bounding box if the model is at least 25% certain that it’s that class. The `line_width` is just the thickness of the bounding boxes.

   Your predictions are located in the folder like `runs/detect/predict`.

5. Upload model to roboflow

   To upload your model to roboflow just replace the parameters with the parameters from the download model code. Once the model is uploaded, you will be able to see the performance of the model in the versions section on roboflow.
