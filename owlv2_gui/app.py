from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from PIL import Image, ImageDraw
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import nms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['RESULT_FOLDER'] = 'results'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="green", width=2)
        draw.text((box[0], box[1]), f"Label: {label}", fill="green")

def process_image(image_path, processor, model, text, threshold, nms_threshold):
    image = Image.open(image_path).convert("RGB")

    # Provide text descriptions for the image
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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        text = request.form.get("text", default="fire smoke")
        threshold = float(request.form.get("threshold", default=0.1))
        nms_threshold = float(request.form.get("nms_threshold", default=0.2))

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            result_path = os.path.join(app.config["RESULT_FOLDER"], file.filename)

            file.save(file_path)

            # Load OwlV2 processor
            processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            # Load OwlV2 object detection model
            model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

            # Process the image and draw bounding boxes
            annotated_image = process_image(file_path, processor, model, text, threshold, nms_threshold)

            # Save the annotated image
            annotated_image.save(result_path)
            print(file.filename)
            return render_template('index.html', filename=file.filename)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RESULT_FOLDER']):
        os.makedirs(app.config['RESULT_FOLDER'])

    app.run(debug=True)
