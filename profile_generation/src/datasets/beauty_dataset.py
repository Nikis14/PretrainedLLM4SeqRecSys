import os
import ast
from typing import List
import pandas as pd
from src.datasets.dataset import Dataset, User, Item
from src.utils import preprocess_null_field


class BeautyItem(Item):
    def __init__(
        self, item_id: str, title: str, description: str, categories: List[str]
    ):
        self.item_id = item_id
        self.title = title
        self.description = description
        self.categories = categories


class BeautyDataset(Dataset):
    def __init__(self, dataset_type: str = "train"):
        super().__init__()
        self.dataset_type = dataset_type

    def load_data(self, folder: str):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder '{folder}' does not exist")

        users_df = pd.read_csv(os.path.join(folder, "raw/users.csv"))

        self.users = {
            str(user.UserId): User(str(user.UserId)) for _, user in users_df.iterrows()
        }

        items_df = pd.read_csv(os.path.join(folder, "raw/items.csv"))

        metadata = []
        with open(os.path.join(folder, "raw/meta_Beauty.json"), "r") as f:
            for line in f.readlines():
                metadata.append(ast.literal_eval(line.strip()))
        metadata_df = pd.DataFrame(metadata)
        items_df = items_df.merge(
            metadata_df, left_on="ProductId", right_on="asin"
        ).drop("asin", axis=1)

        self.items = {}
        for _, item in items_df.iterrows():
            self.items[str(item.ProductId)] = BeautyItem(
                str(item.ProductId),
                preprocess_null_field(item.title),
                preprocess_null_field(item.description),
                item.categories[0],
            )

        interactions_df = pd.read_csv(os.path.join(folder, "raw/interactions.csv"))
        interactions_df = interactions_df.groupby("UserId").filter(
            lambda x: len(x) >= 5
        )
        inter2rating = {}
        for user_id, item_id, rating in zip(
            interactions_df.UserId,
            interactions_df.ProductId,
            interactions_df.Rating,
        ):
            inter2rating[(str(user_id), str(item_id))] = rating

        if self.dataset_type == "train":
            interactions_train = pd.read_pickle(
                os.path.join(folder, "processed/train_sequences.pkl")
            )
        elif self.dataset_type == "val":
            interactions_train = pd.read_pickle(
                os.path.join(folder, "processed/valid_sequences.pkl")
            )
        else:
            interactions_train = pd.read_pickle(
                os.path.join(folder, f"processed_splits/{self.dataset_type}")
            )

        inv_mappings_users, inv_mappings_items = pd.read_pickle(
            os.path.join(folder, "processed/mappings.pkl")
        )
        mappings_users = {str(v): str(k) for k, v in inv_mappings_users.items()}
        mappings_items = {str(v): str(k) for k, v in inv_mappings_items.items()}

        for user_id, items in interactions_train.items():
            user_id = str(mappings_users[str(user_id)])
            for item in items:
                if str(item) in mappings_items:
                    item_id = str(mappings_items[str(item)])
                    rating = inter2rating[user_id, item_id]

                    if user_id not in self.interactions:
                        self.interactions[str(user_id)] = {"ratings": {}}
                    self.interactions[str(user_id)]["ratings"][item_id] = rating

    def get_interactions_by_user(self, user_id):
        return self.interactions.get(str(user_id), {"ratings": {}})
