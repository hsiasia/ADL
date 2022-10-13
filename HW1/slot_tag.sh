# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script

# sh slot_tag.sh data/slot/test.json pred.slot.csv
python test_slot.py --test_file "${1}" --ckpt_path best-slot.pth --pred_file "${2}"