import cv2 as cv
import os
import torch
import random
from datasets.constants import SYMBOLS
from datasets.data_utils import text_to_image,text_to_tensor_and_pad_with_zeros, image_processing_for_real_data
from torch.utils.data import IterableDataset

UNITS = ["mg", "m", "km", "in", "ft", "yd", "mi", "kg", "g", "lb", "oz", "s", "min",
         "h", "d", "wk", "mo", "yr", "L", "mL", "gal", "qt", "pt", "cm", "ha", "ac",
         "K", "J", "cal", "kcal", "kWh", "W", "hp", "Pa", "atm", "bar", "psi", "b",
         "B", "kB", "MB", "GB", "TB"]

class StreamData(IterableDataset):
    def __init__(self):
        self.word_list = open(os.path.join("datasets/LinesData", "words-english-2.txt")).read().splitlines()
        self.real_data = self.real_data_list()

    def generate(self):
        while True:
            image_arr, phrase = self.one_iteration()
            expected = text_to_tensor_and_pad_with_zeros(phrase)
            yield {"image": image_arr, "expected": expected}
    
    def __iter__(self):
        return iter(self.generate())
    
    def one_iteration(self):
        use_real_data = random.randint(1, 100)
        if use_real_data <= 10:
            return self.real_image_and_phrase()
        else:
            return self.generated_image_and_phrase()
    
    def real_data_list(self, folder="datasets/LinesData", text_file="lines.txt"):
        lines = open(os.path.join(folder, text_file)).read().splitlines()

        data_list_dict = []
        for line in lines:
            if line[0] == "#":
                continue
            split_line = line.split()

            if split_line[1] == "ok":
                image_path = split_line[0] + ".png"
                phrase = " ".join(split_line[8::]).replace("|", " ")
                
                image_path_split = image_path.split("-")
                folder_name_1 = image_path_split[0]
                folder_name_2 = "-".join(image_path_split[:2])
                image_file_path = os.path.join(folder, folder_name_1, folder_name_2, image_path)
                image_to_np_array = cv.imread(image_file_path)
                normalized_image = image_processing_for_real_data(image_to_np_array)
                sample = torch.from_numpy(normalized_image).unsqueeze(0), phrase

                data_list_dict.append(sample)
    
        return data_list_dict
    
    def real_image_and_phrase(self):
        selected_idx = random.randint(0, len(self.real_data)-1)
        return self.real_data[selected_idx]
    
    def generated_image_and_phrase(self):
        phrase = self.generate_one_phrase()
        image = text_to_image(phrase)[1]
        return image, phrase

    def generate_one_phrase(self):
        total_length = random.randint(1, 40)
        pulled_phrase = self.word_list[random.randint(0, len(self.word_list)-1)]
        should_space = random.randint(0, 1)

        current_length = 0
        while current_length < total_length:
            phrase_element = self.get_one_phrase_element()
            if should_space:
               pulled_phrase += " " + phrase_element
            else:
                pulled_phrase += "" + phrase_element
            current_length += len(phrase_element)
        
        return pulled_phrase[:total_length]
    
    def get_one_phrase_element(self):
        element_type = random.randint(1, 50)
        use_capitalize = random.randint(0, 1)
        
        if element_type < 30:
            selected_idx = random.randint(0, len(self.word_list)-1)
            return self.word_list[selected_idx].capitalize() if use_capitalize else self.word_list[selected_idx]
        elif element_type < 40:
            return self.generate_random_number_string()
        else:
            selected_idx = random.randint(0, len(SYMBOLS)-1)
            return SYMBOLS[selected_idx]
        
    def generate_random_number_string(self):
        number_length = random.randint(1, 11)
        random_number = random.randint(0, 10**number_length)

        if_use_units = random.randint(0, 1)
        include_commas = random.randint(0, 1)

        if include_commas == 1:
            return f'{random_number:,}'
        else:
            return str(random_number) + UNITS[random.randint(0, len(UNITS)-1)] + " " if if_use_units else ""