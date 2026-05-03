This repository offers implementation of code using Python 3.6

## Installation

```.bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For [NeuSpell](https://github.com/neuspell/neuspell) Python=3.7 was used.

```.bash
pip install -r requirements_neuspell.txt
```

## Datasets

All the public GEC datasets can be downloaded from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data).  
Synthetically PIE created datasets can be generated/downloaded [here](https://github.com/awasthiabhijeet/PIE/tree/master/errorify).  
Spelling Errors [here](https://titan.dcs.bbk.ac.uk/~roger/corpora.html).  

Alternatively  

```.bash
pip install gec-datasets
```

## Spelling Noise

To add spelling noise:

```.bash
python myutils/spelling_noise.py \
    -i INPUT \
    -o OUTPUT \
    -c SPELLNOISE.csv
```

We have added spelling noise to Stage 1 data of [PIE](https://github.com/awasthiabhijeet/PIE/tree/master/errorify) SOURCE part of train and dev

## Preprocessing

To train the model, data has to be preprocessed and converted to a special format:

```.bash
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```

Stage 1 data is very large so a multiprocessing script can be used.

```.bash
python myutils/preprocess_data_faster.py -s SOURCE -t TARGET -o OUTPUT_FILE
```

### Preprocessing with our tags

morphological inflection tags

OUTPUT_FILE from preprocessing_data.py

```.bash
python myutils/process_inflect.py INPUT OUTPUT
```

spelling tags

OUTPUT from process_inflect.py

```.bash
python myutils/process_spelling.py INPUT OUTPUT
```

spelling tags with NeuSpell. It is computationally expensive, and for this, it was given less priority  
OUTPUT from process_spelling.py

```.bash
python myutils/process_spelling_neuspell.py INPUT OUTPUT
```

## Train model

To train the model, simply run:

```.bash
python train.py --train_set TRAIN_SET \
                --dev_set DEV_SET \
                --model_dir MODEL_DIR
```

There are different parameters for training:

* **max_len** maximum sequence length for tokens
* **cold_steps_count** number of epochs where we train only last linear layer
* **n_epoch** training epochs (different for each stage)
* **patience** early stopping
* **transformer_model** model encoder {bert, roberta, xlnet}
* **tp_prob** probability of getting sentences with no errors; helps to balance precision/recall

## Model inference

```.bash
python predict.py --model_path MODEL_PATH [MODEL_PATH ...] \
                  --vocab_path VOCAB_PATH \
                  --input_file INPUT_FILE \
                  --output_file OUTPUT_FILE \
                  -- special_tokens_fix {0 or 1 based on pre-trained model} 
```

### Inference Tweaking

*`min_error_probability` - minimum error probability  
*`additional_confidence` - confidence bias
