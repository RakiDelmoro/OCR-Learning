import string
import random

GENERATED_IMAGE_PARAGRAPH_SIZE = (random.randint(100, 120), random.randint(700, 1000))
GENERATED_IMAGE_SENTENCE_SIZE = (random.randint(100, 120), random.randint(400, 700))
GENERATED_IMAGE_WORD_SIZE = (random.randint(100, 120), random.randint(100, 300))

INPUT_IMAGE_SIZE = (128, 1350)

LETTERS = [ch for ch in string.ascii_letters]
NUMBERS = [num for num in string.digits]
SYMBOLS = ["#", "+", "/", "*", ")", "(", '"', "-", "!", "?", ",", ".", ":", ";", "'", "&"]
START_TOKEN = '\N{Start of Text}'
END_TOKEN = '\N{End of Text}'
PAD_TOKEN = '\N{Substitute}'
CHARS = ["\x00", PAD_TOKEN, START_TOKEN, END_TOKEN] + SYMBOLS + [" "] + LETTERS + NUMBERS

MAX_PHRASE_LENGTH = 93
