import os
import sys
from tqdm import tqdm
from src.utils import read_text, save_json


def main(folder_read, file_save):
    res = {}
    for file in tqdm(os.listdir(folder_read)):
        profile = read_text(os.path.join(folder_read, file))
        user_id = file.rsplit(".", 1)[0]
        res[user_id] = profile
    save_json(res, file_save)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
