print("Initializing segmentation model...")

# ----------------- CONFIGURATIONS ----------------- #
MODEL_DIR   = "ALL_LANG_DATA/Colombia/Bancolombia/model_output_col_multitag"  # Directory where your trained model and tokenizer are saved
TEXT_COLUMN = "text"  # Column name in the CSV that contains the conversation text
WINDOW_SIZE = 14                         # Number of words per sliding window
STRIDE = 6                               # Step size for the sliding window
MAX_LENGTH = 32                          # Maximum token length


