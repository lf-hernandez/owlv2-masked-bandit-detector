from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw
import torch

print("Loading model...")

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("raccoon.jpg")
if image.mode != "RGB":
    image = image.convert("RGB")

text_queries = ["a raccoon", "raccoon"]

inputs = processor(text=text_queries, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.Tensor([image.size[::-1]]).to(device)
results = processor.post_process_grounded_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1
)

print(f"Found {len(results[0]['scores'])} potential raccoons!")

result = image.copy()
draw = ImageDraw.Draw(result)

for score, box in zip(results[0]["scores"], results[0]["boxes"]):
    x1, y1, x2, y2 = box.tolist()
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1 - 20), f"{score:.2f}", fill="red")

result.save("detected_raccoon.jpg")

print("Result saved as 'detected_raccoon.jpg'")
