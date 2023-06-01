import torch
from transformers import Mask2FormerForUniversalSegmentation

def load_Mask2Former_model_from_huggingface(numClass : int) -> torch.nn.Module:
    return Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic", 
                                                            id2label = {idx: '{idx}' for idx in range(numClass)},
                                                            ignore_mismatched_sizes=True)
    

     