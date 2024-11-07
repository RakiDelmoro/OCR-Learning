import os
from main import main
from constants import GREEN, CLEAR
from datasets.data_utils import split_dataset
from constants import NUM_PHRASE_LENGTH, BATCH_SIZE
from torch.utils.data import ConcatDataset
from datasets.dataset import FileTextLineDataset, GeneratedDataset, MedicalNamesDataset

INFERENCE_TEST = ["copy1.png", "test-img-prep3.png"]

DATASET_FILE_1 = FileTextLineDataset("datasets/LinesData", "lines.txt")
DATASET_FILE_2 = GeneratedDataset(number_of_generated_dataset=BATCH_SIZE*200)
DATASET_FILE_3 = MedicalNamesDataset("datasets/LinesData", "medical-names-2.txt")

TRAINING_DATASET, VALIDATION_DATASET = split_dataset(DATASET_FILE_1, split_ratio=0.99)
TRAINING = ConcatDataset([DATASET_FILE_2, TRAINING_DATASET, DATASET_FILE_3])

training_dataset_size = len(TRAINING)
validation_dataset_size = len(VALIDATION_DATASET)
inference_dataset_size = len(INFERENCE_TEST)

main(TRAINING, VALIDATION_DATASET, INFERENCE_TEST,
     training_dataset_size, validation_dataset_size, inference_dataset_size, True)
