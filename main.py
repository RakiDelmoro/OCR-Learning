import os
import torch
import math
from torch.utils.data import DataLoader
from model.model import TransformerOCR
from datasets.data_utils import decode_for_print, collate_function
from constants import BATCH_SIZE, DEVICE, LEARNING_RATE, NUM_EPOCHS, RED, CLEAR, GREEN, WHITE_UNDERLINE
from utils import train_model, evaluate_model, generate_token_prediction, save_checkpoint, load_checkpoint, beam_search_for_inference

# File where to save model checkpoint and fully trained
MODEL_CHECKPOINT_FOLDER = "ModelCheckpoints"
FULLY_TRAINED_MODEL_FILE = "model.pth"

MODEL = TransformerOCR().to(DEVICE)
OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=LEARNING_RATE)
SHCEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode="min", patience=5)

def main(training_dataset, validation_dataset, inference_iterable, training_data_length, validation_data_length,
         inference_data_length, use_checkpoint):

    training_loader = DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    # validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_function)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, drop_last=True)
    
    train_loader_length = training_data_length // training_loader.batch_size
    validation_loader_length = validation_data_length // validation_loader.batch_size

    start_epoch = 1
    if use_checkpoint:
        if os.path.exists(MODEL_CHECKPOINT_FOLDER):
            list_of_checkpoints = os.listdir(MODEL_CHECKPOINT_FOLDER)
            if len(list_of_checkpoints) == 0:
                print(f"No checkpoint file yet.")
            else:
                loaded_epoch, t_loss, v_loss, checkpoint = load_checkpoint(model_checkpoint_folder=MODEL_CHECKPOINT_FOLDER, model=MODEL,
                                                            optimizer=OPTIMIZER)
                print(f"{WHITE_UNDERLINE}loaded checkpoint file from {checkpoint} have EPOCH: {loaded_epoch} with a training loss: {t_loss}, Validation loss: {v_loss}{CLEAR}")
                start_epoch = loaded_epoch + 1

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_model(train_dataset=training_loader, model=MODEL, optimizer=OPTIMIZER,
                                            dataloader_length=train_loader_length, encoder_trainer=False)
        
        val_loss = evaluate_model(validation_dataset=validation_loader, model=MODEL,
                                             dataloader_length=validation_loader_length, encoder_trainer=False)
        
        if math.isnan(train_loss): break 
        SHCEDULER.step(val_loss)

        print(f"{WHITE_UNDERLINE}EPOCH: {epoch} Training loss: {train_loss}, Validation loss: {val_loss}{CLEAR}")        
        for _ in range(inference_data_length):
            predicted, expected = beam_search_for_inference(next(inference_iterable)[1], MODEL)
            print(f"Predicted: {decode_for_print(predicted[0])} Expected: {decode_for_print(expected)}")

        current_validation_loss = 0.0
        if use_checkpoint:
            if val_loss < current_validation_loss:
                checkpoint_file = f"./BestModel/checkpoint.tar"
                current_validation_loss = val_loss
                torch.save(MODEL.state_dict(), checkpoint_file)
                print(f"New Best model Save! {GREEN}{checkpoint_file}{CLEAR}")
            
            print(f"{RED}Saving do not turn off!{CLEAR}")
            save_checkpoint(epoch=epoch, model=MODEL, optimizer=OPTIMIZER,
                                                    t_loss=train_loss, v_loss=val_loss, checkpoint_folder=MODEL_CHECKPOINT_FOLDER)
            print(f"{GREEN}Done saving!{CLEAR}")
    
    torch.save(MODEL.state_dict(), FULLY_TRAINED_MODEL_FILE)
