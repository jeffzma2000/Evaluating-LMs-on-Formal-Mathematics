import os
import sys


"""
Helper script that takes a directory and writes all the directory names to a file.
Use:
    python data_helper.py <dir_name> <output_file>
"""


root = sys.argv[1]
dirs = [x + "\n" for x in os.listdir(root)]
with open(sys.argv[2], "w") as f:
    f.writelines(dirs)
