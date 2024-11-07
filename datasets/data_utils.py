from os import scandir
import cv2 as cv
import random
import numpy as np
import torch
from itertools import takewhile

from feature_flags import USE_DYNAMIC_LENGTH
from torch.utils.data import random_split
from torch.utils.data.dataloader import default_collate
from PIL import Image, ImageDraw, ImageFont
from datasets.constants import CHARS, END_TOKEN, GENERATED_IMAGE_SENTENCE_SIZE, PAD_TOKEN, START_TOKEN, MAX_PHRASE_LENGTH, INPUT_IMAGE_SIZE, SYMBOLS, NUMBERS, LETTERS, GENERATED_IMAGE_WORD_SIZE, GENERATED_IMAGE_PARAGRAPH_SIZE

max_height_by_font_cache = {}

def noise_adder(image_tensor):
    light_noise = torch.rand(128, 1350) * 0.5 + 0.5
    dark_noise = torch.rand(128, 1350) * 0.5
    new_image_as_tensor = torch.zeros((128, 1350))
    light_condition = image_tensor == 1.0
    new_image_as_tensor[light_condition] = light_noise[light_condition]
    dark_condition = image_tensor == 0.0
    new_image_as_tensor[dark_condition] = dark_noise[dark_condition]
    return new_image_as_tensor

def generated_data_processing(image_numpy_array):
    # background_condition = image_numpy_array > 200
    text_condition = image_numpy_array < 130
    # light_noise = np.random.randint(128, 256, size=image_numpy_array.shape).astype(np.uint8)
    dark_noise = np.random.choice([0, 255], size=image_numpy_array.shape, p=[0.8, 0.2]).astype(np.uint8)
    # image_numpy_array[background_condition] += light_noise[background_condition]
    image_numpy_array[text_condition] += dark_noise[text_condition]
    image_arr = np.clip(image_numpy_array, 0, 255).astype(np.uint8)
    return image_arr

def real_image_pixel_manipulation(image_numpy_array):
    average_pixel = np.average(image_numpy_array) - 25
    background_condition = image_numpy_array > average_pixel
    image_numpy_array[background_condition] = 255
    return image_numpy_array

def inference_data_processing(image_arr):
    image_arr = cv.GaussianBlur(image_arr, (3, 3), 0)
    light_condition = image_arr > np.average(image_arr) - 10
    dark_condition = image_arr < np.average(image_arr) - 10
    image_arr[light_condition] = 255
    image_arr[dark_condition] = 0
    image = cv.dilate(image_arr, np.ones((2,2), np.uint8), iterations=1)
    return cv.bitwise_not(image)

def image_processing_for_generated_data(image_as_array):
    image_gray = cv.cvtColor(image_as_array, cv.COLOR_RGB2GRAY)
    rescaled_img = rescale(image_gray, INPUT_IMAGE_SIZE)
    rgb = rescaled_img.min()
    center = (rescaled_img.shape[1] // 2, image_gray.shape[0] // 2)
    angle = random.randint(-2, 2)
    rotate_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotate_img = cv.warpAffine(rescaled_img, rotate_matrix, (rescaled_img.shape[1], rescaled_img.shape[0]), borderValue=int(rgb))
    kernel = np.ones((2,2), np.uint8)
    opening = cv.morphologyEx(rotate_img, cv.MORPH_OPEN, kernel)
    normalized_image = cv.normalize(rotate_img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    return normalized_image

def image_processing_for_real_data(image_as_array):
    image_gray = cv.cvtColor(image_as_array, cv.COLOR_BGR2GRAY)
    image = real_image_pixel_manipulation(image_gray)
    rescaled_img = rescale(image, INPUT_IMAGE_SIZE)
    rgb = rescaled_img.max()
    center = (rescaled_img.shape[1] // 2, rescaled_img.shape[0] // 2)
    angle = random.randint(-1, 1)
    rotate_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotate_img = cv.warpAffine(rescaled_img, rotate_matrix, (rescaled_img.shape[1], rescaled_img.shape[0]), borderValue=int(rgb))
    inverted_img = cv.bitwise_not(rotate_img)
    normalized_image = cv.normalize(inverted_img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    return normalized_image

def apply_augmentation_for_inference(image_as_array):
    image_gray = cv.cvtColor(image_as_array, cv.COLOR_BGR2GRAY)
    rescaled_image = rescale(image_gray, INPUT_IMAGE_SIZE) 
    blur = cv.GaussianBlur(rescaled_image, (3, 3), 0) 
    image_manipulated = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 311, 10)
    normalized_image = cv.normalize(image_manipulated, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    return normalized_image

def get_printable(character):
    if character == PAD_TOKEN: return "🔴"
    if character == START_TOKEN: return "🚦"
    if character == END_TOKEN: return "🤚"
    return character

def collate_function(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

def split_dataset(dataset, split_ratio):
    training_size = int(split_ratio * len(dataset))
    validation_size = int(len(dataset) - training_size)
    return random_split(dataset, [training_size, validation_size])
    
def generate_random_characters(length_of_char: int):
    if USE_DYNAMIC_LENGTH:
        generate = "".join(random.choice(CHARS[4:]) for _ in range(random.randint(1, length_of_char)))
    else:
        generate = "".join(random.choice(CHARS[4:]) for _ in range(length_of_char))
    return generate

def rescale(image, size):
    """
    image: Array
    size: tuple
    """
    image = cv.resize(image, (size[1], size[0]))
    canvas = np.ones(size, dtype=np.uint8) * 255
    canvas[0:size[0], 0:size[1]] = image
    return canvas

def text_to_image(text):
    folder = scandir("./datasets/.fonts")
    font_names = (file.name for file in folder if file.is_file())
    font_name = random.choice(list(font_names))
    max_height_and_font = max_height_by_font_cache.get(font_name)
    background = 0
    text_color = 255, 255, 255
    if len(text) < 15:
        image_size = GENERATED_IMAGE_WORD_SIZE
    elif len(text) < 60:
        image_size = GENERATED_IMAGE_SENTENCE_SIZE
    else:
        image_size = GENERATED_IMAGE_PARAGRAPH_SIZE
    if max_height_and_font != None:
        max_height, font = max_height_and_font
        font_size = font.size
    else:
        font_size = 0
        max_height = 0
        while max_height <= image_size[0]:
            font_size = font_size + 1
            font = ImageFont.truetype(f"datasets/.fonts/{font_name}", font_size)
            _, top, _, bottom = font.getbbox(''.join(CHARS))
            max_height = bottom - top
        max_height_by_font_cache[font_name] = max_height, font
        font_size = font_size -1
        font = ImageFont.truetype(f"datasets/.fonts/{font_name}", font_size)
    width = font.getlength(text)
    while width > image_size[1]:
        font_size = font_size - 1
        font = ImageFont.truetype(f"datasets/.fonts/{font_name}", font_size)
        width = font.getlength(text)
    image = Image.new("RGB", (image_size[1], image_size[0]), color=background)
    draw = ImageDraw.Draw(image)
    _, top, _, bottom = font.getbbox(text)
    draw.text((image_size[1]/2, image_size[0]/2), text, anchor="mm", fill=text_color, font=font)
    image_np_arr = np.asarray(image)
    normalized_image = image_processing_for_generated_data(image_np_arr)
    return image, normalized_image

def text_to_tensor_and_pad_with_zeros(text, max_length=MAX_PHRASE_LENGTH):
    num_padding = max_length - len(text)
    label_tokens_for_decoder = [char_to_index[START_TOKEN]] + encode(text) + [char_to_index[END_TOKEN]] + [char_to_index[PAD_TOKEN]]
    label_tokens_for_encoder = encode(text)
    if num_padding != 0:
        label_tokens_for_decoder.extend([char_to_index[PAD_TOKEN]] * num_padding)
        label_tokens_for_encoder.extend([char_to_index[PAD_TOKEN]] * num_padding)
        # if num_padding == 0 else + [char_to_index[PAD_TOKEN] for _ in range(num_padding)]
    encoder_label = torch.tensor(label_tokens_for_encoder, dtype=torch.long)
    decoder_label = torch.tensor(label_tokens_for_decoder, dtype=torch.long)
    return encoder_label, decoder_label

int_to_printable = [get_printable(c) for _, c in enumerate(CHARS)]
char_to_index = {c:i for i, c in enumerate(CHARS)}
encode = lambda text: [char_to_index[c] for c in text]
decode_for_print = lambda tensor: "".join([int_to_printable[int(each_token)]
                                           for each_token in takewhile(lambda x: x != char_to_index[PAD_TOKEN], tensor)])
