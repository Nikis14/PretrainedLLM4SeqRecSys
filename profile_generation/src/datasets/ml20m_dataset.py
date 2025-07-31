import os
import re
from typing import List
import pandas as pd
from src.datasets.dataset import Dataset, User, Item
from src.utils import preprocess_null_field


class MovieLensItem(Item):
    def __init__(
        self,
        item_id: str,
        release_year: str,
        title: str,
        genres: List[str],
    ):
        self.item_id = item_id
        self.release_year = release_year
        self.title = title
        self.genres = genres


def extract_from_last_parentheses_regex(text):
    try:
        match = re.findall(r"\((.*?)\)", text)
        if match:
            return match[-1]
        else:
            return None
    except (AttributeError, TypeError):
        return None


class MovieLensDataset(Dataset):
    def __init__(self, dataset_type: str = "train"):
        super().__init__()
        self.dataset_type = dataset_type

    def load_data(self, folder: str):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder '{folder}' does not exist")
        items_df = pd.read_csv(os.path.join(folder, "raw/movies.csv"))
        self.items = {}
        for _, item in items_df.iterrows():
            self.items[str(item.movieId)] = MovieLensItem(
                item_id=str(item.movieId),
                release_year=preprocess_null_field(
                    extract_from_last_parentheses_regex(item.title)
                ),
                title=preprocess_null_field(item.title),
                genres=preprocess_null_field(item.genres).split("|"),
            )

        interactions_df = pd.read_csv(os.path.join(folder, "raw/ratings.csv"))
        inter2rating = {}
        for user_id, item_id, rating in zip(
            interactions_df.userId,
            interactions_df.movieId,
            interactions_df.rating,
        ):
            inter2rating[(str(user_id), str(item_id))] = rating
            self.users[str(user_id)] = User(str(user_id))
        if self.dataset_type == "train":
            interactions_train = pd.read_pickle(
                os.path.join(folder, "processed/train_sequences.pkl")
            )
        elif self.dataset_type == "val":
            interactions_train = pd.read_pickle(
                os.path.join(folder, f"processed/valid_sequences.pkl")
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
        return self.interactions.get(user_id, {"ratings": {}})
