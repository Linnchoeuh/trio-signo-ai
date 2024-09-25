# Trio-Signo AI-Service

LSF AI model recognition

# Setup

Install python 3.12 or higher and run:
```sh
pip install -r requirements.txt
```

# Launch API
```sh
python main.py
```
For details about the option do:
```sh
python main.py --help
```

# Add image to the dataset

You can either manually add them in the folder `datasets/source_images/{whatever_foldze_you_want}/your_image.png`

**or**

run:
```sh
python example.py
```
Do a sign and press on the corresponding letter on the keyboard or `0`if not a sign

# Generate dataset
> Find info with `-h` parameter
```sh
python dataset_asm.py
```

# Train AI Model

```sh
python TrainLSFAlphabetRecognizer.py
```

# Test AI Model

```sh
python example.py
```
Do a si
