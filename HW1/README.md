# Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python train_intent.py
```

## Intent detection
### train
```shell
# you can add arg as --data_dir <data_dir>... after the basic command
python train_intent.py
```

### predict
```shell
# as reproduce
python test_intent.py --ckpt_path <ckpt_path>
```

### reproduce my result
```shell
bash download.sh
bash intent_cls.sh data/intent/test.json pred.intentest.csv
```

---

## Slot tagging
### train
```shell
# you can add arg as --data_dir <data_dir>... after the basic command
python train_slot.py
```

### predict
```shell
# as reproduce
python test_slot.py --ckpt_path <ckpt_path>
```

### reproduce result
```shell
bash download.sh
bash slot_tag.sh data/slot/test.json pred.slotest.csv
```