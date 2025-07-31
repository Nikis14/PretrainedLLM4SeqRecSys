from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class User(ABC):
    def __init__(self, id: str):
        self.id = id
        self.interactions: Dict[str, Any] = {}

    def get_interactions(self) -> Dict[str, Any]:
        return self.interactions


class Item(ABC):
    def __init__(self, item_id: str, name: str):
        self.item_id = item_id
        self.name = name


class Dataset(ABC):
    def __init__(self):
        self.users = {}
        self.items = {}
        self.interactions = {}

    @abstractmethod
    def load_data(self, folder: str):
        pass

    def get_users(self, user_ids: Optional[list] = None) -> List[User]:
        if user_ids is None:
            return list(self.users.values())
        return [
            self.users.get(user_id) for user_id in user_ids if user_id in self.users
        ]

    def get_items(self) -> List[Item]:
        return list(self.items.values())

    @abstractmethod
    def get_interactions_by_user(self, user_id: str) -> Dict[str, Any]:
        pass
