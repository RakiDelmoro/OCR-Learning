import os
import torch
from PIL import Image
from main import MODEL_CHECKPOINT_FOLDER
from utils import decode_for_print
from model.model import TransformerOCR
from datasets.data_utils import text_to_image
from utils import load_checkpoint_for_test, get_latest_save_checkpoint, beam_search_for_test
from constants import GREEN, CLEAR, RED

with torch.inference_mode():
    model = TransformerOCR().to("cuda")
    model_checkpoint = "mark-2.tar"
    model.load_state_dict(torch.load(os.path.join("ModelCheckpoints", model_checkpoint))["model_state_dict"])
    # model.load_state_dict(torch.load(os.path.join("BestModel", "checkpoint.tar")))

    predicted = beam_search_for_test("test-img-prep3.png", model, beam_power=5)
    print(f"{RED}{model_checkpoint}{CLEAR}")
    print(f"{GREEN}{decode_for_print(predicted)}{CLEAR}")
