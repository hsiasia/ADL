# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script

# sh intent_cls.sh data/intent/test.json pred.intent.csv
python test_intent.py --test_file "${1}" --ckpt_path best-intent.pth --pred_file "${2}"