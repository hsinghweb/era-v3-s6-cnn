import torch
from src.model import Net

def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be < 20000"

def test_model_components():
    model = Net()
    
    # Test for Batch Normalization
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use Batch Normalization"
    
    # Test for Dropout
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"
    
    # Test for GAP or FC
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_gap or has_fc, "Model should use either GAP or FC layer" 