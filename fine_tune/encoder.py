"""
Code from https://github.com/sthalles/SimCLR/blob/master/models/resnet_simclr.py
"""

import torch.nn as nn
import torchvision.models as models
import torch
from transformers import AutoModelForImageClassification, AutoConfig
import torch.nn.functional as F

class InvalidBackboneError():
    """Raised when the choice of backbone Convnet is invalid."""

class FeatureEncoder(nn.Module):

    def __init__(self,):
        super(FeatureEncoder, self).__init__()
        
        model_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
        
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Replace the last layer 

        mlp_layers = [
            nn.Linear(self.config.hidden_size, 256), 
            nn.ReLU(),
        ]

        self.model.classifier = nn.Sequential(*(mlp_layers))


    def forward(self, x):
        # return self.backbone(x)
        
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic', antialias=True)
        model = self.model.to(torch.device("cuda"))
        return model(x).logits
    
    
class Classifier(nn.Module):

    def __init__(self,):
        super(Classifier, self).__init__()
        
        model_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
        
        self.model = AutoModelForImageClassification.from_pretrained(model_name)


    def forward(self, x):
        # return self.backbone(x)
        
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic', antialias=True)
        model = self.model.cuda()
        
        logits = model(x).logits
        probabilities = F.softmax(logits, dim=1)

        _, predicted_class = torch.max(probabilities, dim=1)
    
        return predicted_class
    
    