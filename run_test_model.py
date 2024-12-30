import os
import torch
from PIL import Image
from main import MODEL_CHECKPOINT_FOLDER
from utils import decode_for_print
from model.model import TransformerOCR
from datasets.data_utils import text_to_image
from utils import load_checkpoint_for_test, get_latest_save_checkpoint, beam_search_for_test, beam_search_for_inference_previous_version
from constants_v1 import GREEN, CLEAR, RED

with torch.inference_mode():
    model = TransformerOCR().to("cuda")
    model_checkpoint = "mark-0.tar"
    model.load_state_dict(torch.load(os.path.join("ModelCheckpoints2", model_checkpoint))["model_state_dict"])
    # model.load_state_dict(torch.load(os.path.join("BestModel", "checkpoint.tar")))

    predicted = beam_search_for_inference_previous_version("copy1.png", model)
    # predicted = beam_search_for_test("copy1.png", model, beam_power=5)
    print(f"Model Use: {RED}{model_checkpoint}{CLEAR}")
    print(f"Model Prediction: {GREEN}{decode_for_print(predicted)}{CLEAR}")
