import os 
import torch, torchvision

import platonic
from datasets import load_dataset
from measure_alignment import compute_score, prepare_features


class Alignment():

    def __init__(self, dataset, subset, models=[], transform=None, device="cuda", dtype=torch.float32):
        
        self.dataset = dataset
            
        # if subset not in platonic.SUPPORTED_DATASETS:
        #     raise ValueError(f"subset {subset} not supported for dataset {dataset}")
        
        self.models = models
        self.device = device
        self.dtype = dtype

        return
    

    def load_features(self, feat_path):
        """ loads features for a model """
        return torch.load(feat_path, map_location=self.device)["feats"].to(dtype=self.dtype)

    
    def get_data(self):

        return self.dataset
    
    def score(self, features1, features2, metric, *args, **kwargs):
        """ 
        Args:
            features (torch.Tensor): features to compare
            metric (str): metric to use
            *args: additional arguments for compute_score / metrics.AlignmentMetrics
            **kwargs: additional keyword arguments for compute_score / metrics.AlignmentMetrics
        Returns:
            dict: scores for each model organized as 
                {model_name: (score, layer_indices)} 
                layer_indices are the index of the layer with maximal alignment
        """
        scores = {}
        for m in self.models:
            scores[m] = compute_score(
                prepare_features(features1, exact=True).to(device=self.device, dtype=self.dtype), 
                prepare_features(features2, exact=True).to(device=self.device, dtype=self.dtype) ,
                metric, 
                *args, 
                **kwargs
            )
        return scores        
    
    