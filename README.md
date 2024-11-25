# MNIST Classification

![Build Status](https://github.com/hsinghweb/era-v3-s6-cnn/actions/workflows/ml-pipeline.yml/badge.svg)

A PyTorch implementation of MNIST digit classification achieving 99.4% test accuracy with less than 20k parameters.

## Features
- Batch Normalization
- Dropout
- Global Average Pooling
- Less than 20k parameters
- 99.4% test accuracy

## Requirements
- Python 3.8+
- PyTorch 1.7+
- See requirements.txt for full list

## Model Training Logs (from GitHub Actions https://github.com/hsinghweb/era-v3-s6-cnn/actions/runs/12004858232/job/33460499642)

```
Total Model Parameters: 10,550
Dataset Split:
Training samples: 50,000
Validation/Test samples: 10,000
Split ratio: 50000/10000
Epoch 1: Test set: Average loss: 0.0963, Accuracy: 97.14%
Epoch 2: Test set: Average loss: 0.0647, Accuracy: 97.90%
Epoch 3: Test set: Average loss: 0.0491, Accuracy: 98.53%
Epoch 4: Test set: Average loss: 0.0447, Accuracy: 98.64%
Epoch 5: Test set: Average loss: 0.0304, Accuracy: 99.08%
Epoch 6: Test set: Average loss: 0.0391, Accuracy: 98.70%
Epoch 7: Test set: Average loss: 0.0399, Accuracy: 98.62%
Epoch 8: Test set: Average loss: 0.0248, Accuracy: 99.16%
Epoch 9: Test set: Average loss: 0.0413, Accuracy: 98.69%
Epoch 10: Test set: Average loss: 0.0292, Accuracy: 99.07%
Epoch 11: Test set: Average loss: 0.0225, Accuracy: 99.31%
Epoch 12: Test set: Average loss: 0.0210, Accuracy: 99.30%
Epoch 13: Test set: Average loss: 0.0270, Accuracy: 99.11%
Epoch 14: Test set: Average loss: 0.0230, Accuracy: 99.19%
Epoch 15: Test set: Average loss: 0.0185, Accuracy: 99.39%
Epoch 16: Test set: Average loss: 0.0167, Accuracy: 99.46%
Reached target accuracy of 99.4% at epoch 16
Training Complete!
==================================================
Dataset Split Summary:
Training Set: 50,000 samples
Validation/Test Set: 10,000 samples
Split Ratio: 50000/10000
--------------------------------------------------
Total Model Parameters: 10,550
Best Validation/Test Accuracy: 99.46%
Final Training Loss: 0.0291
Final Validation/Test Loss: 0.0167
Training stopped at epoch: 16/19
==================================================
```



## Model Validation Logs (from GitHub Actions https://github.com/hsinghweb/era-v3-s6-cnn/actions/runs/12004858232/job/33460499642)

```
============================= test session starts ==============================
platform linux -- Python 3.8.18, pytest-8.3.3, pluggy-1.5.0
rootdir: /home/runner/work/era-v3-s6-cnn/era-v3-s6-cnn
collected 2 items
src/test_model.py 
Model Parameter Count Test:
Total parameters in model: 10,550
Parameter limit: 20,000
Loaded trained model weights successfully
Current model accuracy: 99.46%
Required accuracy: 99.4%
.
Model Components Test:
Has Batch Normalization layers: True
Has Dropout layers: True
Has Global Average Pooling: True
Has Fully Connected layer: True
Model Architecture:
conv1: Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
bn1: BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
conv2: Conv2d(8, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
bn2: BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
pool1: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
dropout1: Dropout(p=0.1, inplace=False)
conv3: Conv2d(12, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
bn3: BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
pool2: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
dropout2: Dropout(p=0.1, inplace=False)
conv4_1x1: Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
conv4_main: Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
bn4: BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
conv5: Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
bn5: BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
attention: Sequential(
  (0): AdaptiveAvgPool2d(output_size=1)
  (1): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
  (2): ReLU()
  (3): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
  (4): Sigmoid()
)
conv6: Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
bn6: BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
dropout3: Dropout(p=0.1, inplace=False)
gap: AdaptiveAvgPool2d(output_size=1)
fc: Linear(in_features=16, out_features=10, bias=True)
.
=============================== warnings summary ===============================
src/test_model.py::test_parameter_count
  /home/runner/work/era-v3-s6-cnn/era-v3-s6-cnn/src/test_model.py:18: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    model.load_state_dict(torch.load('best_model.pth'))
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 2 passed, 1 warning in 4.24s =========================
```