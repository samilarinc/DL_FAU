import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

Trainer.restore_checkpoint(0, '../last/')
Trainer.save_onnx('last_model.onnx')