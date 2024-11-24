import torch
from model import Net

def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameter Count Test:")
    print(f"Total parameters in model: {total_params:,}")
    print(f"Parameter limit: 20,000")
    assert total_params < 20000, f"Model has {total_params:,} parameters, should be < 20,000"

def test_model_components():
    model = Net()
    print("\nModel Components Test:")
    
    # Test for Batch Normalization
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    print(f"Has Batch Normalization layers: {has_bn}")
    assert has_bn, "Model should use Batch Normalization"
    
    # Test for Dropout
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    print(f"Has Dropout layers: {has_dropout}")
    assert has_dropout, "Model should use Dropout"
    
    # Test for GAP or FC
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    print(f"Has Global Average Pooling: {has_gap}")
    print(f"Has Fully Connected layer: {has_fc}")
    assert has_gap or has_fc, "Model should use either GAP or FC layer"

    # Print model architecture summary
    print("\nModel Architecture:")
    for name, module in model.named_children():
        print(f"{name}: {module}")