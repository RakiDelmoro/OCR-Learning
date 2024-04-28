import os
from main import main
from constants import GREEN, CLEAR
from datasets.data_utils import split_dataset
from constants import NUM_PHRASE_LENGTH 
from torch.utils.data import ConcatDataset
from datasets.dataset import FileTextLineDataset, GeneratedDataset

DATASET_FILE_1 = FileTextLineDataset("datasets/LinesData", "lines.txt")
DATASET_FILE_2 = GeneratedDataset()

TRAINING_DATASET, VALIDATION_DATASET = split_dataset(DATASET_FILE_1, split_ratio=0.9)
VALIDATION_SPLIT, TEST_SPLIT = split_dataset(VALIDATION_DATASET, split_ratio=0.9)
TRAINING = ConcatDataset([DATASET_FILE_2, TRAINING_DATASET])

training_dataset_size = 64*100 + len(TRAINING_DATASET)
validation_dataset_size = len(VALIDATION_SPLIT)
inference_dataset_size = 5

main(TRAINING, VALIDATION_SPLIT, enumerate(TEST_SPLIT),
     training_dataset_size, validation_dataset_size, inference_dataset_size, True)
