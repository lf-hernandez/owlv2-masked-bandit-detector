import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw
from typing import List


class Detector:
    def __init__(self):
        print("Loading OWLv2 model...")
        self.processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def detect(
        self,
        image: Image.Image,
        text_queries: List[str] = None,
        threshold: float = 0.1
    ) -> List[dict]:
        if text_queries is None:
            text_queries = ["a raccoon", "raccoon"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(
            text=text_queries,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run detection
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results to get bounding boxes
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )

        # Format detections
        detections = []
        for score, box in zip(results[0]["scores"], results[0]["boxes"]):
            detections.append({
                "score": float(score),
                "box": box.tolist()  # [x1, y1, x2, y2]
            })

        return detections

    def detect_and_draw(
        self,
        image: Image.Image,
        text_queries: List[str] = None,
        threshold: float = 0.1
    ) -> Image.Image:
        detections = self.detect(image, text_queries, threshold)

        result = image.copy()
        draw = ImageDraw.Draw(result)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 20), f"{det['score']:.2f}", fill="red")

        return result
