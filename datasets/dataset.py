import os
import torch
import random
import cv2 as cv
from iterable_dataset import UNITS
from torch.utils.data import Dataset
from datasets.constants import MAX_PHRASE_LENGTH, SYMBOLS
from datasets.data_utils import text_to_tensor_and_pad_with_zeros, text_to_image, image_processing_for_real_data

class RealTextLineDataset(Dataset):
    def __init__(self, dataset_file, txt_file):
        super().__init__()
        self.dataset_folder = dataset_file
        self.txt_file = txt_file  
        lines = open(os.path.join(self.dataset_folder, self.txt_file)).read().splitlines()
        dataset_dict = list()
        for line in lines:
            if line[0] == "#": continue
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
    def __init__(self, number_of_generated_dataset: int):
        super().__init__()
        self.words = open(os.path.join("datasets/LinesData", "words-english-2.txt")).read().splitlines()
        self.length_dataset = number_of_generated_dataset

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, _):
        phrase = self.generate_one_phrase()
        image = text_to_image(phrase)[1]
        expected = text_to_tensor_and_pad_with_zeros(phrase)
        sample = {"image": torch.from_numpy(image).unsqueeze(0), "expected": expected}
        return sample

    def generate_one_phrase(self):
        total_phrase_length = random.choice([24, 26])
        should_space = random.randint(0, 1)
        captilized = random.randint(0, 1)
        generated_phrase = self.words[random.randint(0, len(self.words)-1)].capitalize() if captilized else self.words[random.randint(0, len(self.words)-1)]
        generated_phrase_length = len(generated_phrase)
        while generated_phrase_length < total_phrase_length:
            random_capitalize = random.randint(0, 2)
            if random_capitalize == 0: phrase_element = self.get_one_phrase_element()
            elif random_capitalize == 1: phrase_element = self.get_one_phrase_element().capitalize()
            else: phrase_element = self.get_one_phrase_element().upper()
            if should_space: generated_phrase += " " + phrase_element
            else: generated_phrase += "" + phrase_element
            generated_phrase_length += len(phrase_element)
        return generated_phrase[:total_phrase_length]

    def get_one_phrase_element(self):
        element_type = random.randint(1, 100)
        if element_type < 20:
            selected_idx = random.randint(0, len(self.words)-1)             
            pulled_phrase = self.words[selected_idx] # Fetch Word
            return pulled_phrase
        elif element_type < 60:
            return self.generate_random_number_string() # Fetch Number
        else:
            selected_idx = random.randint(0, len(SYMBOLS)-1)
            return SYMBOLS[selected_idx] # Fetch symbol

    def generate_random_number_string(self):
        number_length = random.randint(1, 11)
        random_number = random.randint(0, 10**number_length)
        if_use_units = random.randint(0, 1)
        include_commas = random.randint(0, 1)
        if include_commas: return f'{random_number:,}' + UNITS[random.randint(0, len(UNITS)-1)] + " " if if_use_units else ""
        else: return  str(random_number) + UNITS[random.randint(0, len(UNITS)-1)] + " " if if_use_units else ""

class MedicalNamesDataset(Dataset):
    def __init__(self, dataset_folder, txt_file):
        super().__init__()
        self.names = open(os.path.join(dataset_folder, txt_file)).read().splitlines()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, each):
        medical_name = self.names[each]
        medical_name_as_image = text_to_image(medical_name)[1]
        expected = text_to_tensor_and_pad_with_zeros(medical_name)
        sample = {"image": torch.from_numpy(medical_name_as_image).unsqueeze(0), "expected": expected}
        return sample
