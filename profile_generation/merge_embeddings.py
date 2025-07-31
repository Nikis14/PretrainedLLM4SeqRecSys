import os
import sys
from tqdm import tqdm
from src.utils import read_json, save_json


def main(folder_read, start_i, end_i, file_save):
    res = {}
    for ind in tqdm(range(int(start_i), int(end_i) + 1)):
        data = read_json(
            os.path.join(folder_read, f"embeddings_{ind}.json")
        )
        res.update(data)
    print("CNT TOTAL:", len(res))
    save_json(res, file_save)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
