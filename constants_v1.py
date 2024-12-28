import torch
import string
from model.model import TransformerOCR


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (64, 768)

NUM_EPOCHS = 100000000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# File where to save model checkpoint and fully trained
MODEL_CHECKPOINT_FOLDER = "ModelCheckpoints2"
FULL_TRAINED_MODEL = "model.pth"

START_TOKEN = '\N{Start of Text}'
END_TOKEN = '\N{End of Text}'
PAD_TOKEN = '\N{Substitute}'

MAX_PHRASE_LENGTH = 93
NUM_PHRASE_LENGTH = 20
INFERENCE_PHRASE_LENGTH = 93

RED = "\033[0;31m"
CLEAR = '\033[0m'
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
WHITE_UNDERLINE = "\033[4;37m"
