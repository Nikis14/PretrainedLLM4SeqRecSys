import os
import pandas as pd
from src.datasets.dataset import Dataset, User, Item
from src.utils import preprocess_null_field


class KionItem(Item):
    def __init__(
        self,
        item_id: str,
        name: str,
        release_year: str,
        genres: str,
        countries: str,
        keywords: str,
        directors_translated,
    ):
        self.item_id = item_id
        self.name = name
        self.release_year = release_year
        self.genres = genres
        self.countries = countries
        self.keywords = keywords
        self.directors_translated = directors_translated


class KionDataset(Dataset):
    def load_data(self, folder: str):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder '{folder}' does not exist")
        users_df = pd.read_csv(os.path.join(folder, "raw/users_en.csv"))
        self.users = {
            str(user.user_id): User(str(user.user_id))
            for _, user in users_df.iterrows()
        }

        items_df = pd.read_csv(os.path.join(folder, "raw/items_en.csv"))
        self.items = {}
        for _, item in items_df.iterrows():
            self.items[str(item.item_id)] = KionItem(
                item_id=str(item.item_id),
                name=preprocess_null_field(item.title),
                release_year=preprocess_null_field(item.release_year),
                genres=preprocess_null_field(item.genres),
                countries=preprocess_null_field(item.countries),
                keywords=preprocess_null_field(item.keywords),
                directors_translated=preprocess_null_field(item.directors_translated),
            )

        interactions_df = pd.read_csv(os.path.join(folder, "raw/interactions.csv"))
        inter2watch_time = {}
        for user_id, item_id, watched_pct in zip(
            interactions_df.user_id,
            interactions_df.item_id,
            interactions_df.watched_pct,
        ):
            inter2watch_time[(str(user_id), str(item_id))] = watched_pct

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
                    watch_percent = inter2watch_time[user_id, item_id]

                    if user_id not in self.interactions:
                        self.interactions[str(user_id)] = {"watched %": {}}
                    self.interactions[str(user_id)]["watched %"][item_id] = (
                        watch_percent
                    )

    def get_interactions_by_user(self, user_id):
        return self.interactions.get(user_id, {"ratings": {}})
