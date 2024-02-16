<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill SigLIP Module

This repository contains the code supporting the SigLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[CLIP](https://github.com/openai/CLIP), developed by OpenAI, is a computer vision model trained using pairs of images and text. You can use CLIP with autodistill for image classification.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [SigLIP Autodistill documentation](https://autodistill.github.io/autodistill/base_models/siglip/).

## Installation

To use SigLIP with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-clip
```

## Quickstart

```python
from autodistill_siglip import SigLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our SigLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
labels = ["person", "a forklift"]
base_model = SigLIP(
    ontology=CaptionOntology({item: item for item in labels})
)

results = base_model.predict("image.jpeg", confidence=0.1)

top_1 = results.get_top_k(1)

# show top label
print(labels[top_1[0][0]])

# label folder of images
base_model.label("./context_images", extension=".jpeg")
```


## License

The SigLIP model is licensed under an [Apache 2.0 license](https://huggingface.co/google/siglip-base-patch16-224).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!