import torch

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 19
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 