from iterable_dataset import StreamData
from main import main
from constants import INFERENCE_PHRASE_LENGTH, NUM_PHRASE_LENGTH
TRAINING_DATASET_SIZE = 64*200
VALIDATION_DATASET_SIZE = 64*10
INFERENCE_DATASET_SIZE = 5

training_dataset = StreamData()
validation_dataset = StreamData()
inference_dataset = StreamData()

main(training_dataset, validation_dataset, enumerate(inference_dataset), TRAINING_DATASET_SIZE, VALIDATION_DATASET_SIZE, INFERENCE_DATASET_SIZE, True)
