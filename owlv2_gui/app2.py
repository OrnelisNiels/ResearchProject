from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw
from torchvision.ops import nms
import torch
import gradio as gr
import numpy as np
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load OwlV2 processor
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
# Load OwlV2 object detection model
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

def convert_to_yolo_format(image_size, box, label):
    # Convert bounding box coordinates to YOLO format
    xmin, ymin, xmax, ymax = box
    width, height = image_size
    x_center = (xmin + xmax) / 2 / width
    y_center = (ymin + ymax) / 2 / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"{label} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)

    boxes = predictions[0]['boxes'].detach().cpu().numpy()
    labels = predictions[0]['labels'].detach().cpu().numpy()

    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="green", width=3)
        draw.text((box[0], box[1]), f"Label: {label}", fill="green")

def process_image(image, processor, model, text, nms_threshold):
    image = Image.open(image).convert("RGB")
    texts = [text.split()]  # Split the user-provided text into a list of words

    # Tokenize and process inputs
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

    # Get raw model outputs
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)

    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)

    # Apply NMS thresholding to the results
    for result in results:
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
        keep = nms(boxes, scores, nms_threshold)
        result["boxes"] = boxes[keep]
        result["scores"] = scores[keep]
        result["labels"] = labels[keep]

    # Draw bounding boxes on the image
    draw_boxes(image, results)

    return image

def object_detection(image, Prompt="fire, smoke", NMS_threshold=0.2):
    if isinstance(image, np.ndarray):
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)

    # Save the image to an in-memory file-like object
    image_io = io.BytesIO()
    image.save(image_io, format='JPEG')
    image_io.seek(0)

    # Process the image and draw bounding boxes
    annotated_image = process_image(image_io, processor, model, Prompt, NMS_threshold)

    # Convert to Gradio format
    annotated_image = np.array(annotated_image)

    return annotated_image

iface = gr.Interface(
    fn=object_detection,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Textbox(label="Prompt", value="fire, smoke"),
        gr.Slider(label="NMS Threshold", value=0.1, step=0.01, maximum=1, minimum=0),
    ],
    outputs=gr.Image(label="Annotated Image"),
)

iface.launch()