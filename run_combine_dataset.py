import torch
from main import runner
from model.model import TransformerOCR
from torch.utils.data import ConcatDataset
from datasets.data_utils import split_dataset
from constants_v1 import BATCH_SIZE, DEVICE, LEARNING_RATE
from datasets.dataset import RealTextLineDataset, GeneratedDataset, MedicalNamesDataset

# Model Properties
MODEL = TransformerOCR().to(DEVICE)
OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=LEARNING_RATE)
SHCEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode="min", patience=5)
# Dataset Properties 
INFERENCE_TEST = ["copy1.png", "test-img-prep3.png"]
SENTENCE_LINE_DATASET = RealTextLineDataset("datasets/LinesData", "lines.txt")
GENERATED_SENTENCE_LINE_DATASET = GeneratedDataset(number_of_generated_dataset=BATCH_SIZE*200)
GENERATED_MEDICAL_DRUG_NAMES = MedicalNamesDataset("datasets/LinesData", "medical-names-2.txt")
TRAINING_DATASET, VALIDATION_DATASET = split_dataset(SENTENCE_LINE_DATASET, split_ratio=0.99)
CONCATENATED_TRAINING_DATASET = ConcatDataset([GENERATED_SENTENCE_LINE_DATASET, TRAINING_DATASET, GENERATED_MEDICAL_DRUG_NAMES])
# Dataset sizes
training_dataset_size = len(CONCATENATED_TRAINING_DATASET)
validation_dataset_size = len(VALIDATION_DATASET)
inference_dataset_size = len(INFERENCE_TEST)

runner(MODEL, OPTIMIZER, SHCEDULER, CONCATENATED_TRAINING_DATASET, VALIDATION_DATASET, INFERENCE_TEST, training_dataset_size, validation_dataset_size, inference_dataset_size, True)
