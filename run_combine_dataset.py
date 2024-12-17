import os
from main import main
from constants import GREEN, CLEAR
from datasets.data_utils import split_dataset
from constants import NUM_PHRASE_LENGTH, BATCH_SIZE
from torch.utils.data import ConcatDataset
from datasets.dataset import FileTextLineDataset, GeneratedDataset, MedicalNamesDataset

INFERENCE_TEST = ["copy1.png", "test-img-prep3.png"]

SENTENCE_LINE_DATASET = FileTextLineDataset("datasets/LinesData", "lines.txt")
GENERATED_SENTENCE_LINE_DATASET = GeneratedDataset(number_of_generated_dataset=BATCH_SIZE*200)
GENERATED_MEDICAL_DRUG_NAMES = MedicalNamesDataset("datasets/LinesData", "medical-names-2.txt")
TRAINING_DATASET, VALIDATION_DATASET = split_dataset(SENTENCE_LINE_DATASET, split_ratio=0.99)
CONCATENATED_TRAINING_DATASET = ConcatDataset([GENERATED_SENTENCE_LINE_DATASET, TRAINING_DATASET, GENERATED_MEDICAL_DRUG_NAMES])

training_dataset_size = len(CONCATENATED_TRAINING_DATASET)
validation_dataset_size = len(VALIDATION_DATASET)
inference_dataset_size = len(INFERENCE_TEST)

main(CONCATENATED_TRAINING_DATASET, VALIDATION_DATASET, INFERENCE_TEST,
     training_dataset_size, validation_dataset_size, inference_dataset_size, False)
