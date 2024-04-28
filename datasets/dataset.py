import numpy as np
import random
import os
import torch
import cv2 as cv
from torch.utils.data import Dataset, IterableDataset, ChainDataset, DataLoader
from datasets.data_utils import text_to_tensor_and_pad_with_zeros, text_to_image, image_processing_for_generated_data, image_processing_for_real_data
from datasets.constants import MAX_PHRASE_LENGTH, SYMBOLS
from iterable_dataset import UNITS

class FileTextLineDataset(Dataset):
    def __init__(self, dataset_file, txt_file):
        super().__init__()
        self.dataset_folder = dataset_file
        self.txt_file = txt_file
        
        lines = open(os.path.join(self.dataset_folder, self.txt_file)).read().splitlines()
        dataset_dict = list()
        for line in lines:
            if line[0] == "#":
                continue
            split_line = line.split()

            if split_line[1] == "ok":
                image_path = split_line[0] + ".png"
                sentence = " ".join(split_line[8::]).replace("|", " ")
                dataset_dict.append({image_path: sentence})
        
        self.dataset = dataset_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        image_file, target = next(iter(self.dataset[index].items()))
        image_file_split = image_file.split("-")
        folder_name1 = image_file_split[0]
        folder_name2 = "-".join(image_file_split[:2])
        # Get the image file path
        image_file_path = os.path.join(self.dataset_folder, folder_name1, folder_name2, image_file)
        image_to_np_aray = cv.imread(image_file_path)
        normalized_image = image_processing_for_real_data(image_to_np_aray)
        target_as_tensor = text_to_tensor_and_pad_with_zeros(text=target, max_length=MAX_PHRASE_LENGTH)
        
        sample = {"image": torch.from_numpy(normalized_image).unsqueeze(0), "expected": target_as_tensor}

        return sample

class GeneratedDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.words = open(os.path.join("datasets/LinesData", "words-english-2.txt")).read().splitlines()

    def __len__(self):
        return len(self.words)

    def __getitem__(self, _):
        phrase = self.generate_one_phrase()
        image = text_to_image(phrase)[1]
        expected = text_to_tensor_and_pad_with_zeros(phrase)
        
        sample = {"image": torch.from_numpy(image).unsqueeze(0), "expected": expected}

        return sample

    def generate_one_phrase(self):
        total_phrase_length = random.randint(1, 93)
        should_space = random.randint(0, 1)
        captilized = random.randint(0, 1)

        generated_phrase = self.words[random.randint(0, len(self.words)-1)].capitalize() if captilized else self.words[random.randint(0, len(self.words)-1)]
        generated_phrase_length = len(generated_phrase)

        while generated_phrase_length < total_phrase_length:
            random_capitalize = random.randint(0, 2)
            if random_capitalize == 0:
                phrase_element = self.get_one_phrase_element()
            elif random_capitalize == 1:
                phrase_element = self.get_one_phrase_element().capitalize()
            else:
                phrase_element = self.get_one_phrase_element().upper()
            
            if should_space:
                generated_phrase += " " + phrase_element
            else:
                generated_phrase += "" + phrase_element
            generated_phrase_length += len(phrase_element)

        return generated_phrase[:total_phrase_length]

    def get_one_phrase_element(self):
        element_type = random.randint(1, 100)

        if element_type < 80:
            # word
            selected_idx = random.randint(0, len(self.words)-1)
            pulled_phrase = self.words[selected_idx]
            return pulled_phrase
        elif element_type < 95:
            # number
            return self.generate_random_number_string()
        else:
            # symbol
            selected_idx = random.randint(0, len(SYMBOLS)-1)
            return SYMBOLS[selected_idx]
        
    def generate_random_number_string(self):
        number_length = random.randint(1, 11)
        random_number = random.randint(0, 10**number_length)

        if_use_units = random.randint(0, 1)
        include_commas = random.randint(0, 1)

        if include_commas:
            return f'{random_number:,}' + UNITS[random.randint(0, len(UNITS)-1)] + " " if if_use_units else ""
        else:
            return  str(random_number) + UNITS[random.randint(0, len(UNITS)-1)] + " " if if_use_units else ""
        
