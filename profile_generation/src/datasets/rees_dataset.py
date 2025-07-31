import os
import polars as pl
import pandas as pd
from src.datasets.dataset import Dataset, User, Item
from src.utils import preprocess_null_field


class ReesItem(Item):
    def __init__(self, item_id: str, category_code: str, brand: str, price: str):
        self.item_id = item_id
        self.category_code = category_code
        self.brand = brand
        self.price = price


class ReesDataset(Dataset):
    def load_data(self, folder: str):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder '{folder}' does not exist")
        users_df = pl.read_csv(os.path.join(folder, "raw/users.csv"))
        self.users = {
            str(user_id): User(str(user_id)) for user_id, _ in users_df.iter_rows()
        }

        items_df = pl.read_csv(os.path.join(folder, "raw/items.csv"))

        self.items = {}
        for item_id, _, category_code, brand, price in items_df.iter_rows():
            self.items[str(item_id)] = ReesItem(
                str(item_id),
                preprocess_null_field(str(category_code)),
                preprocess_null_field(str(brand)),
                preprocess_null_field(str(price)),
            )

        interactions_df = pl.read_csv(os.path.join(folder, "raw/interactions.csv"))
        inter2event = {}
        for _, user_id, _, item_id, event_type, _ in interactions_df.iter_rows():
            inter2event[(str(user_id), str(item_id))] = event_type

        interactions_train = pd.read_pickle(
            os.path.join(folder, "processed/train_sequences.pkl")
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
                    event = inter2event[user_id, item_id]

                    if user_id not in self.interactions:
                        self.interactions[str(user_id)] = {"event_type": {}}
                    self.interactions[str(user_id)]["event_type"][item_id] = event

    def get_interactions_by_user(self, user_id):
        return self.interactions.get(str(user_id), {"event_type": {}})
