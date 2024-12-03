import os
import torch
import Levenshtein
import cv2 as cv
import numpy as np
import statistics as S
import random
from tqdm import tqdm
from datasets.data_utils import decode_for_print, char_to_index, inference_data_processing, rescale, INPUT_IMAGE_SIZE
from constants import DEVICE, START_TOKEN, END_TOKEN, INFERENCE_PHRASE_LENGTH, BLUE, CYAN, CLEAR
from model.model import HybridLoss

# CONSTANTS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_TOKEN = '\N{Start of Text}'
END_TOKEN = '\N{End of Text}'
PAD_TOKEN = '\N{Substitute}'

hybrid_loss = HybridLoss(lambda_val=0.5)

def train_model(train_dataset, model, optimizer, dataloader_length,
                encoder_trainer):
    model.train()
    print("TRAINING--->>>")
    running_loss = 0.0
    loop = tqdm(enumerate(train_dataset), total=dataloader_length, leave=False)
    for i, each in loop:
        image = each['image'].to(DEVICE)
        expected_target = each['expected']
        if encoder_trainer:decoder_loss, encoder_loss, _, _ = model(image, expected_target)
        else: decoder_loss, encoder_loss, _, _ = model(image, expected_target)
        loss = hybrid_loss(encoder_loss, decoder_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i >= dataloader_length: break
    training_loss = running_loss / dataloader_length
    return training_loss

def evaluate_model(validation_dataset, model, dataloader_length,
                   encoder_trainer):

    print("EVALUATING--->>>")
    model.eval()

    batch_distance_accumulator = []
    running_loss = 0.0
    
    loop = tqdm(enumerate(validation_dataset), total=dataloader_length, leave=False)
    with torch.no_grad():
        for i, each in loop:
            image = each["image"].to(DEVICE)
            expected_target = each["expected"]

            if encoder_trainer:
                decoder_loss, encoder_loss, logits, expected_output = model(image, expected_target)
            
            else:
                decoder_loss, encoder_loss, logits, expected_output = model(image, expected_target)

            loss = hybrid_loss(encoder_loss, decoder_loss)
            batch_distance_and_string = each_batch_distance_and_string(logits, expected_output)
            batch_distance_accumulator.extend(batch_distance_and_string)
            
            predicted_and_expected_list = model_prediction_and_expected_to_string(logits, expected_output)

            running_loss += loss.item()
            if i >= dataloader_length: break
    
    list_of_distance = [distance for each_dict in batch_distance_accumulator for distance in each_dict.keys()]
    highest_distance = sorted(list_of_distance, reverse=True)[:5]
    minimum_distance = sorted(list_of_distance)[:5]
    
    print(f"{CYAN}Minimum model prediction{CLEAR}")
    for each in batch_distance_accumulator:
        counter = 0
        string_distance, model_and_expected_str = next(iter(each.items()))
        if string_distance in minimum_distance:
            print(f"Predicted: {model_and_expected_str[0]}, Expected: {model_and_expected_str[1]}")
            counter+=1
        
        if counter >= len(minimum_distance): break

    print(f"{CYAN}Maximum model prediction{CLEAR}")
    for each in batch_distance_accumulator:
        counter = 0
        string_distance, model_and_expected_str = next(iter(each.items()))
        if string_distance in highest_distance:
            print(f"Predicted: {model_and_expected_str[0]}, Expected: {model_and_expected_str[1]}")
            counter+=1
        
        if counter >= len(highest_distance): break

    percentile_10 = np.percentile(list_of_distance, 10)
    percentile_90 = np.percentile(list_of_distance, 90)

    validation_loss = running_loss / dataloader_length
    
    print(f"{BLUE}Print out model prediction{CLEAR}")
    number_to_print = random.randint(10, len(predicted_and_expected_list))
    for each in range(number_to_print):
        predicted_and_expected = predicted_and_expected_list[each]
        print(f"Predicted: {predicted_and_expected[0]} Target: {predicted_and_expected[1]}")

    print(f"Percentile 10: {percentile_10}, Percentile 90: {percentile_90}")
    print(f"Average: {S.fmean(list_of_distance)}, Minimum: {min(list_of_distance)}, Maximum: {max(list_of_distance)}")

    return validation_loss

def save_checkpoint(epoch, model, optimizer, t_loss, v_loss, checkpoint_folder):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": t_loss,
        "val_loss": v_loss
    }

    torch.save(checkpoint, f"./{checkpoint_folder}/mark-{epoch%5}.tar")

def get_latest_save_checkpoint(model_checkpoint_folder, list_of_checkpoint_files): 
    latest_modified_checkpoint_file = ""
    for each in range(len(list_of_checkpoint_files)):
        checkpoint_file = list_of_checkpoint_files[each]

        if latest_modified_checkpoint_file == "":
            latest_modified_checkpoint_file += checkpoint_file

        checkpoint_status = os.path.getmtime(os.path.join(model_checkpoint_folder, checkpoint_file))
        latest_checkpoint_status = os.path.getmtime(os.path.join(model_checkpoint_folder, latest_modified_checkpoint_file))

        if checkpoint_status > latest_checkpoint_status:
            latest_modified_checkpoint_file = ""
            latest_modified_checkpoint_file += checkpoint_file

    return os.path.join(model_checkpoint_folder, latest_modified_checkpoint_file)

def load_checkpoint(model_checkpoint_folder, model, optimizer):
    list_of_file_checkpoints = os.listdir(model_checkpoint_folder)
    for _ in range(len(list_of_file_checkpoints)):
        try:
            checkpoint_file = get_latest_save_checkpoint(model_checkpoint_folder, list_of_file_checkpoints)
            # checkpoint_file = os.path.join(model_checkpoint_folder, "checkpoint_v2-2.tar")
            checkpoint_file_loaded = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint_file_loaded["model_state_dict"])
            optimizer.load_state_dict(checkpoint_file_loaded["optimizer_state_dict"])
            load_epoch = checkpoint_file_loaded["epoch"]
            train_loss = checkpoint_file_loaded["train_loss"]
            val_loss = checkpoint_file_loaded["val_loss"]
            print(f"Successfully load checkpoint from file: {checkpoint_file}")
            break
        except RuntimeError:
            print(f"Failed to load checkpoint from file: {checkpoint_file}")
            list_of_file_checkpoints.remove(checkpoint_file)
            continue

    return load_epoch, train_loss, val_loss, checkpoint_file

def load_checkpoint_for_test(model_checkpoint_folder, model):
    list_of_file_checkpoints = os.listdir(model_checkpoint_folder)
    for _ in range(len(list_of_file_checkpoints)):
        try:
            checkpoint_file = get_latest_save_checkpoint(model_checkpoint_folder, list_of_file_checkpoints)
            checkpoint_file_loaded = torch.load(checkpoint_file)
            model_state_dict = model.load_state_dict(checkpoint_file_loaded["model_state_dict"])
        except RuntimeError:
            print(f"Failed to load checkpoint from file: {checkpoint_file}")
            list_of_file_checkpoints.remove(checkpoint_file)
            continue

    return model_state_dict

def model_prediction_and_expected_to_string(batched_model_logits, batched_expected):
    logits_data = batched_model_logits.data
    high_probability_char = logits_data.topk(1)[1].squeeze(-1)
    
    batched_model_prediction = high_probability_char.cpu().numpy()

    model_prediction_and_expected_list = []
    for i in range(batched_model_logits.shape[0]):
        predicted_sequence = batched_model_prediction[i]
        expected_sequence = batched_expected[i]
        model_prediction_as_string, expected_as_string = decode_for_print(predicted_sequence), decode_for_print(expected_sequence)

        model_prediction_and_expected_list.append((model_prediction_as_string, expected_as_string))

    return model_prediction_and_expected_list

def calculate_distance_with_corresponding_string(model_output, expected):
    model_pred = model_output.data
    high_idx = model_pred.topk(1)[1].squeeze(-1)
    model_pred = high_idx.cpu().numpy()
    
    model_output_str, expected_str = decode_for_print(model_pred), decode_for_print(expected)
    distance = Levenshtein.distance(model_output_str, expected_str)

    max_length = max(len(model_output_str), len(expected_str))
    similarity = float(max_length - distance) / float(max_length)

    return similarity, (model_output_str, expected_str)

def each_batch_distance_and_string(batch_model_output, batch_expected):
    each_distance_and_string = []
    batch_iter = batch_model_output.shape[0]
    for each in range(batch_iter):
        each_model_and_expected = batch_model_output[each], batch_expected[each]
        distance, model_and_expected_string = calculate_distance_with_corresponding_string(each_model_and_expected[0], each_model_and_expected[1])
        
        each_distance_and_string.append({distance: model_and_expected_string})

    return each_distance_and_string

def generate_square_mask(size):
    mask = (torch.triu(torch.ones((size, size), device=DEVICE)) == 1).transpose(1, 0)
    mask = mask.float().masked_fill(mask == 0, float("-inf")
                                     ).masked_fill(mask == 1, float(0.0)).to(DEVICE)
    return mask

def generate_token_prediction(model_input, model, max_length=INFERENCE_PHRASE_LENGTH+3):
    image = model_input['image'].unsqueeze(0).to("cuda")
    expected = model_input['expected'][1].to("cuda")
    model.eval()

    # image = model_input.unsqueeze(0).to("cuda")
    memory = model.encode(image).to("cuda")
    predicted = torch.tensor([char_to_index[START_TOKEN]], device="cuda", dtype=torch.long).unsqueeze(0)  #torch.ones(1, 1).fill_(char_to_index[START_TOKEN]).type(torch.long).to("cuda")
    for _ in range(max_length-1):
        start_masked = generate_square_mask(predicted.size(1))
        memory_mask = torch.ones((memory.shape[0], memory.shape[1]), device="cuda")
        out = model.decode(predicted, memory, start_masked, memory_mask)
        prob = out[:, -1]
        _, next_char_predicted = torch.max(prob, dim=1)
        next_char_predicted = next_char_predicted.item()

        predicted = torch.cat((predicted, torch.tensor([next_char_predicted], device=DEVICE).unsqueeze(0)), dim=1)

        if next_char_predicted == char_to_index[END_TOKEN]:
            break

    return predicted, expected

def beam_search_for_inference(model_input, model, max_length=INFERENCE_PHRASE_LENGTH+3):
    model.eval()
    image = model_input['image'].unsqueeze(0).to("cuda")
    expected = model_input['expected'][1].to("cuda")
    # image_as_array = cv.imread(image_file)

    memory = model.encode(image)
    beam = [(torch.tensor([char_to_index[START_TOKEN]], device="cuda", dtype=torch.long).unsqueeze(0), 0)]
    for _ in range(max_length-1):
        new_beam = []
        for tokens, score in beam:
            if tokens[:, -1] == char_to_index[END_TOKEN]:    
                new_beam.append((tokens, score))
                continue
            
            start_masked = generate_square_mask(tokens.size(1))
            memory_mask = torch.ones((memory.shape[0], memory.shape[1]), device="cuda")
            out = model.decode(tokens, memory, start_masked, memory_mask)
            prob = torch.nn.functional.softmax(out[:, -1], dim=1)
            probability, token = torch.topk(prob, 1)

            for i in range(1):
                next_char_index = token[0][i].item()
                next_char_probability = probability[0][i].item()

                new_tokens = torch.cat((tokens, torch.tensor([next_char_index], device="cuda").unsqueeze(0)), dim=1)
                new_score = score + next_char_probability
                new_beam.append((new_tokens, new_score))
    
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:1]

    best_sequence, best_score = beam[0]
    return best_sequence, expected

def beam_search_for_inference_previous_version(image, model, max_length=INFERENCE_PHRASE_LENGTH+3):
    model.eval()
    image_as_array = cv.imread(image)
    grey_image = cv.cvtColor(image_as_array, cv.COLOR_BGR2GRAY)
    image_manipulated = inference_data_processing(grey_image)
    
    image = rescale(image_manipulated, INPUT_IMAGE_SIZE)

    norm_image = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    image_input = torch.from_numpy(norm_image).unsqueeze(0).to("cuda")
    
    memory = model.encode(image_input.unsqueeze(0))
    beam = [(torch.tensor([char_to_index[START_TOKEN]], device="cuda", dtype=torch.long).unsqueeze(0), 0)]
    
    for _ in range(max_length-1):
        new_beam = []
        for tokens, score in beam:
            if tokens[:, -1] == char_to_index[END_TOKEN]:    
                new_beam.append((tokens, score))
                continue
            
            start_masked = generate_square_mask(tokens.size(1))
            memory_mask = torch.ones((memory.shape[0], memory.shape[1]), device="cuda")
            out = model.decode(tokens, memory, start_masked, memory_mask)
            prob = torch.nn.functional.softmax(out[:, -1], dim=1)
            probability, token = torch.topk(prob, 3)

            for i in range(3):
                next_char_index = token[0][i].item()
                next_char_probability = probability[0][i].item()

                new_tokens = torch.cat((tokens, torch.tensor([next_char_index], device="cuda").unsqueeze(0)), dim=1)
                new_score = score + next_char_probability
                new_beam.append((new_tokens, new_score))
    
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:3]

    best_sequence, best_score = beam[0]
    return best_sequence[0]

def beam_search_for_test(image_file, model, beam_power=3, max_length=INFERENCE_PHRASE_LENGTH+3):
    model.eval()
    batch_size = 2**beam_power
    image_as_array = cv.imread(image_file)
    grey_image = cv.cvtColor(image_as_array, cv.COLOR_BGR2GRAY)
    image_manipulated = inference_data_processing(grey_image)
    
    image = rescale(image_manipulated, INPUT_IMAGE_SIZE)

    norm_image = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    image_input = torch.from_numpy(norm_image).unsqueeze(0).to("cuda")
    
    memory = model.encode(image_input.repeat(batch_size, 1, 1, 1))
    next_input = torch.tensor([char_to_index[START_TOKEN]], device="cuda", dtype=torch.long).repeat(batch_size, 1)
    
    for i in range(1, max_length):
        start_masked = generate_square_mask(i)
        memory_mask = torch.ones((memory.shape[0], memory.shape[1]), device="cuda")
        model_output = model.decode(next_input, memory, start_masked, memory_mask)
        normalized_output = torch.nn.functional.softmax(model_output, dim=-1)
        
        average_one_hots = torch.mean(normalized_output, 0, False)
        highest_predicted_character_for_each_position = torch.topk(average_one_hots, 1)
        
        worst_characters_index = torch.topk(highest_predicted_character_for_each_position.values.squeeze(-1), min(i, beam_power), largest=False).indices
        next_input = highest_predicted_character_for_each_position.indices.squeeze(-1)
        
        for j in range(0, beam_power):
            next_input = next_input.repeat(2, 1)
            character_position_to_change = worst_characters_index[min(j, worst_characters_index.shape[0]-1)]
            top_two_of_character = torch.topk(average_one_hots[character_position_to_change], 2)
            
            for k in range(0, next_input.shape[0]//2):
                next_input[k][character_position_to_change] = top_two_of_character.indices[1]

        next_input = torch.cat((torch.tensor([[2]], device="cuda").repeat(next_input.shape[0], 1), next_input), 1)

    return next_input[0]
