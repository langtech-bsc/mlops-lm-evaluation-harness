import os
from glob import glob

for filename in os.listdir():
    if "esbbq" in filename:
        os.rename(filename, filename.replace("esbbq", "cabbq"))

for new_filename in glob("*.yaml"):
    with open(new_filename, mode="r") as f_in:
        original = f_in.read()
        fixed = original.replace("esbbq", "cabbq")

        with open(new_filename, mode="w+") as f_out:
            f_out.write(fixed)
