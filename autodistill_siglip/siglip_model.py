import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
import numpy as np
from transformers import pipeline

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class SigLIP(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.model = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-224")
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Classifications:
        prompts = self.ontology.prompts()
        outputs = self.model(load_image(input, return_format="PIL"), candidate_labels=prompts)
        outputs = [{"score": round(output["score"], 4), "label": prompts.index(output["label"]) } for output in outputs]

        results = sv.Classifications(
            class_id=np.array([output["label"] for output in outputs if output["score"] > confidence]),
            confidence=np.array([output["score"] for output in outputs if output["score"] > confidence])
        )

        return results