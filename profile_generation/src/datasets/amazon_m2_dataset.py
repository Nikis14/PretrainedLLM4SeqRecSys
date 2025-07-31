import os
import polars as pl
import pandas as pd
from src.datasets.dataset import Dataset, User, Item
from src.utils import preprocess_null_field


class AmazonM2Item(Item):
    def __init__(
        self,
        item_id: str,
        title: str,
        price: float,
        brand: str,
        color: str,
        size: str,
        model: str,
        material: str,
        desc: str,
    ):
        self.item_id = item_id
        self.title = title
        self.price = price
        self.brand = brand
        self.color = color
        self.size = size
        self.model = model
        self.material = material
        self.desc = desc


class AmazonM2Dataset(Dataset):
    def load_data(self, folder: str):
        """
        Loads the AmazonM2 dataset from a folder.

        The folder is expected to contain the following files:

        - products_train.csv: a CSV file containing product information
        - train_sequences.pkl: a pickle file containing user interaction sequences
        - mappings.pkl: a pickle file containing mappings from user/item IDs to integers

        The method populates the `items` and `users` attributes of the dataset instance
        with the loaded data.

        Args:
            folder (str): The path to the folder containing the dataset files.

        Raises:
            FileNotFoundError: If any of the required files are missing.
        """
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder '{folder}' does not exist")

        items_df = pl.read_csv(os.path.join(folder, "products_train.csv"))
        if items_df is None:
            raise FileNotFoundError(
                f"Could not find file 'products_train.csv' in folder '{folder}'"
            )

        self.items = {}
        for (
            item_id,
            _,
            title,
            price,
            brand,
            color,
            size,
            model,
            material,
            _,
            desc,
        ) in items_df.iter_rows():
            if item_id is None:
                continue
            self.items[str(item_id)] = AmazonM2Item(
                str(item_id),
                preprocess_null_field(title),
                preprocess_null_field(price),
                preprocess_null_field(brand),
                preprocess_null_field(color),
                preprocess_null_field(size),
                preprocess_null_field(model),
                preprocess_null_field(material),
                preprocess_null_field(desc),
            )

        if not os.path.exists(os.path.join(folder, "train_sequences.pkl")):
            raise FileNotFoundError(
                f"Could not find file 'train_sequences.pkl' in folder '{folder}'"
            )

        interactions_train = pd.read_pickle(os.path.join(folder, "train_sequences.pkl"))

        if not os.path.exists(os.path.join(folder, "mappings.pkl")):
            raise FileNotFoundError(
                f"Could not find file 'mappings.pkl' in folder '{folder}'"
            )
        inv_mappings_users, inv_mappings_items = pd.read_pickle(
            os.path.join(folder, "mappings.pkl")
        )
        mappings_users = {str(v): str(k) for k, v in inv_mappings_users.items()}
        mappings_items = {str(v): str(k) for k, v in inv_mappings_items.items()}

        users = list(pd.read_pickle(os.path.join(folder, "train_sequences.pkl")).keys())
        self.users = {
            mappings_users[str(user)]: User(mappings_users[str(user)]) for user in users
        }
        for user_id, items in interactions_train.items():
            mapped_user_id = str(mappings_users[str(user_id)])
            for item in items:
                if str(item) in mappings_items:
                    item_id = str(mappings_items[str(item)])

                    if mapped_user_id not in self.interactions:
                        self.interactions[str(mapped_user_id)] = {"interactions": {}}
                    self.interactions[str(mapped_user_id)]["interactions"][item_id] = (
                        "interaction"
                    )

    def get_interactions_by_user(self, user_id):
        return self.interactions.get(str(user_id), {"interactions": {}})
