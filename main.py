import os
import torch
import math
from torch.utils.data import DataLoader
from datasets.data_utils import decode_for_print
from constants_v1 import BATCH_SIZE, NUM_EPOCHS, RED, CLEAR, GREEN, WHITE_UNDERLINE, MODEL_CHECKPOINT_FOLDER, FULL_TRAINED_MODEL
from utils import train_model, evaluate_model, save_checkpoint, beam_search_for_inference_previous_version, check_model_checkpoint_availability, save_best_model

def runner(model, optimizer, scheduler, training_dataset, validation_dataset, inference_iterable, training_data_length, validation_data_length, inference_data_length, use_checkpoint):
    training_loader = DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    train_loader_length = training_data_length // training_loader.batch_size
    validation_loader_length = validation_data_length // validation_loader.batch_size
    start_epoch = 1

    if use_checkpoint: start_epoch = check_model_checkpoint_availability()

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_model(train_dataset=training_loader, model=model, optimizer=optimizer, dataloader_length=train_loader_length, encoder_trainer=False)
        val_loss = evaluate_model(validation_dataset=validation_loader, model=model, dataloader_length=validation_loader_length, encoder_trainer=False)
        if math.isnan(train_loss): break 
        scheduler.step(val_loss)
        print(f"{WHITE_UNDERLINE}EPOCH: {epoch} Training loss: {train_loss}, Validation loss: {val_loss}{CLEAR}")    
        with torch.inference_mode():    
            for each in inference_iterable:
                # predicted, expected = beam_search_for_inference(next(inference_iterable)[1], MODEL)
                predicted = beam_search_for_inference_previous_version(each, model)
                print(f"Predicted: {decode_for_print(predicted)}")
        current_validation_loss = 0.0
        if use_checkpoint:
            if val_loss < current_validation_loss: current_validation_loss = save_best_model(model, val_loss)

            print(f"{RED}Saving do not turn off!{CLEAR}")
            save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, t_loss=train_loss, v_loss=val_loss, checkpoint_folder=MODEL_CHECKPOINT_FOLDER)
            print(f"{GREEN}Done saving!{CLEAR}")
    torch.save(model.state_dict(), FULL_TRAINED_MODEL)
