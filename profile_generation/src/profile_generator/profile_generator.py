import os
from typing import Dict, List, Any, Optional, Tuple
import json
from tqdm import tqdm
from src.datasets.dataset import Dataset
from src.llm import LLM
from src.llm import Embedder
from src.datasets import User
from src.profile_generator.generation_pipelines import UserProfileGenerator


class ProfileGenerator:
    def __init__(
        self, user_profile_gen_pipeline: UserProfileGenerator, batch_size: int = 128
    ):
        self.user_profile_gen_pipeline: UserProfileGenerator = user_profile_gen_pipeline
        self.batch_size = batch_size

    def _prepare_list_users(
        self,
        user_ids: List[str],
        interaction_field: str,
        llm: LLM,
        dataset: Dataset,
        max_output_tokens: Optional[int] = None,
    ):
        """
        Prepares and categorizes users into one-time and several-time users based on their interactions.

        This method processes a list of users, retrieves their interactions from the dataset,
        and enriches their interaction data with additional item metadata. It then determines
        the number of interaction splits required for each user and categorizes them into one-time
        users or several-time users based on the result.

        Args:
            user_ids (List[str]): A list of user IDs to process.
            interaction_field (str): The field in the dataset representing user interactions.
            llm (LLM): The language model used for interaction processing.
            dataset (Dataset): The dataset containing user and item information.
            max_output_tokens (Optional[int]): The maximum number of output tokens for descriptions.

        Returns:
            Tuple[List[Tuple[User, List[Dict[str, Any]]]], List[Tuple[User, List[Dict[str, Any]]]]]:
            A tuple containing two lists:
                - The first list consists of users categorized as one-time users.
                - The second list consists of users categorized as several-time users.
        """

        one_time_users = []
        several_time_users = []

        for user in tqdm(
            dataset.get_users(user_ids), desc="User processing", total=len(user_ids)
        ):
            user_interactions: Dict[str, Any] = dataset.get_interactions_by_user(
                user.id
            )
            if user_interactions is None:
                continue
            user.interactions = user_interactions
            item_id2meta_info = dataset.items

            for field_name in list(item_id2meta_info.values())[0].__dict__:
                if field_name == "item_id":
                    continue
                if field_name == "name":
                    field_name = "title"
                user.interactions[field_name] = {}

            for item_id in user.interactions[interaction_field]:
                for field_name in item_id2meta_info[item_id].__dict__:
                    if field_name == "item_id":
                        continue
                    if field_name == "name":
                        user.interactions["title"][item_id] = item_id2meta_info[
                            item_id
                        ].__dict__["name"]
                    else:
                        user.interactions[field_name][item_id] = item_id2meta_info[
                            item_id
                        ].__dict__[field_name]
            interactions_with_info = prepare_list_of_objects(
                user.interactions, interaction_field
            )

            number_of_splits = self.user_profile_gen_pipeline.check_number_of_splits(
                interactions_with_info, llm, max_output_tokens
            )

            if number_of_splits == 1:
                one_time_users.append((user, interactions_with_info))
            else:
                several_time_users.append((user, interactions_with_info))
        return one_time_users, several_time_users

    def _generate_one_time_users(
        self,
        one_time_users: List[Tuple[User, List[Dict[str, Any]]]],
        descriptions_path: str,
        llm: LLM,
        max_output_tokens: Optional[int],
    ) -> Dict[str, str]:
        descriptions: Dict[str, str] = {}
        for i in tqdm(
            range(0, len(one_time_users), self.batch_size),
            desc="Generating descriptions for one-time users",
        ):
            user_descriptions = (
                self.user_profile_gen_pipeline.generate_description_batch(
                    [
                        interactions_with_info
                        for _, interactions_with_info in one_time_users[
                            i : i + self.batch_size
                        ]
                    ],
                    llm,
                    max_output_tokens,
                )
            )

            for (user, _), user_description in zip(
                one_time_users[i : i + self.batch_size], user_descriptions
            ):
                descriptions[user.id] = user_description
                user.description = descriptions[user.id]

                with open(
                    os.path.join(descriptions_path, str(user.id) + ".txt"),
                    mode="w",
                    encoding="utf-8",
                ) as f:
                    f.write(user_description)

        return descriptions

    def _generate_several_time_users(
        self,
        several_time_users: List[Tuple[User, List[Dict[str, Any]]]],
        descriptions_path: str,
        llm: LLM,
        max_output_tokens: Optional[int],
    ) -> Dict[str, str]:
        descriptions: Dict[str, str] = {}

        for user, interactions_with_info in tqdm(
            several_time_users, desc="Generating descriptions for several-time users"
        ):
            user_description = self.user_profile_gen_pipeline.generate_description(
                interactions_with_info, llm, max_output_tokens
            )

            descriptions[user.id] = user_description
            user.description = descriptions[user.id]

            with open(
                os.path.join(descriptions_path, str(user.id) + ".txt"),
                mode="w",
                encoding="utf-8",
            ) as f:
                f.write(user_description[0])

        return descriptions

    def generate_descriptions(
        self,
        dataset: Dataset,
        llm: LLM,
        max_output_tokens: Optional[int],
        user_ids: List[str],
        descriptions_path: str,
        interaction_field: str,
    ):
        """
        Generate descriptions for users.

        Args:
            dataset: The dataset to use for generating descriptions.
            llm: The language model to use for generating descriptions.
            max_output_tokens: The maximum number of tokens to generate for each
                description.
            user_ids: The list of user IDs to generate descriptions for.
            descriptions_path: The path to the directory where the descriptions
                should be saved.
            interaction_field: The field name in the dataset that contains the
                interactions.

        Returns:
            A dictionary mapping user IDs to their generated descriptions.
        """
        descriptions: Dict[str, str] = {}

        # use descriptions_preprocessed.json to store already splitted users
        condition = os.path.exists(
            os.path.join(descriptions_path, "descriptions_preprocessed.json")
        )
        if not condition:
            one_time_users, several_time_users = self._prepare_list_users(
                user_ids, interaction_field, llm, dataset, max_output_tokens
            )

            with open(
                os.path.join(descriptions_path, "descriptions_preprocessed.json"),
                mode="w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "one_time_users": {
                            user.id: info for user, info in one_time_users
                        },
                        "several_time_users": {
                            user.id: info for user, info in several_time_users
                        },
                    },
                    f,
                    ensure_ascii=False,
                )
        else:
            with open(
                os.path.join(descriptions_path, "descriptions_preprocessed.json"),
                mode="r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
                one_time_users = [
                    (User(user_id), info)
                    for user_id, info in data["one_time_users"].items()
                    if user_id in user_ids
                ]
                several_time_users = [
                    (User(user_id), info)
                    for user_id, info in data["several_time_users"].items()
                    if user_id in user_ids
                ]

        if len(one_time_users) == 0 and len(several_time_users) == 0:
            raise Exception("No users to generate descriptions for")

        one_time_descriptions = self._generate_one_time_users(
            one_time_users, descriptions_path, llm, max_output_tokens
        )

        several_time_descriptions = self._generate_several_time_users(
            several_time_users, descriptions_path, llm, max_output_tokens
        )

        descriptions = {**one_time_descriptions, **several_time_descriptions}

        return descriptions

    @classmethod
    def generate_embeddings(cls, user2descripton: dict, embedder: Embedder):
        embeddings: Dict[str, List[float]] = {}

        print("Start preparing batches...")
        user_id_batches, desc_batches = divide_users_into_batches(user2descripton, 48)
        print("Finished preparing batches...")
        for id_batch, desc_batch in tqdm(list(zip(user_id_batches, desc_batches))):
            emebddings_batch = embedder.get_embeddings(desc_batch)
            for i in range(len(desc_batch)):
                embeddings[id_batch[i]] = emebddings_batch[i]
        return embeddings


class SeveralProfileGenerator(ProfileGenerator):
    def _generate_one_time_users(
        self, one_time_users, descriptions_path, llm, max_output_tokens
    ):
        descriptions: Dict[str, str] = {}
        for i in tqdm(
            range(0, len(one_time_users), self.batch_size),
            desc="Generating descriptions for one-time users",
        ):
            user_descriptions: Dict[str, List[str]] = (
                self.user_profile_gen_pipeline.generate_description_batch(
                    [
                        interactions_with_info
                        for _, interactions_with_info in one_time_users[
                            i : i + self.batch_size
                        ]
                    ],
                    llm,
                    max_output_tokens,
                )
            )
            for user, _ in one_time_users[i : i + self.batch_size]:
                descriptions[user.id] = {}

            for j, prompt_type in enumerate(list(user_descriptions.keys()), 1):
                for (user, _), user_description in zip(
                    one_time_users[i : i + self.batch_size],
                    user_descriptions[prompt_type],
                ):
                    descriptions[user.id][prompt_type] = user_description
                    user.description = descriptions[user.id]

                    with open(
                        os.path.join(
                            os.path.join(descriptions_path, f"type{j}"),
                            str(user.id) + ".txt",
                        ),
                        mode="w",
                        encoding="utf-8",
                    ) as f:
                        f.write(user_description)

        return descriptions

    def _generate_several_time_users(
        self, several_time_users, descriptions_path, llm, max_output_tokens
    ):
        descriptions: Dict[str, str] = {}

        for user, interactions_with_info in tqdm(
            several_time_users, desc="Generating descriptions for several-time users"
        ):
            user_description = self.user_profile_gen_pipeline.generate_description(
                interactions_with_info, llm, max_output_tokens
            )
            descriptions[user.id] = user_description
            user.description = descriptions[user.id]

            for i, prompt_type in enumerate(list(user_description.keys()), 1):
                with open(
                    os.path.join(
                        os.path.join(descriptions_path, f"type{i}"),
                        str(user.id) + ".txt",
                    ),
                    mode="w",
                    encoding="utf-8",
                ) as f:
                    f.write(user_description[prompt_type][0])

        return descriptions


def prepare_list_of_objects(
    interactions: Dict[str, Any], interaction_field: str
) -> List[Dict[str, Any]]:
    res = []
    for item in interactions[interaction_field]:
        cur_dct = {}
        for field in interactions:
            cur_dct[field] = interactions[field][item]
        res.append(cur_dct)
    return res


def divide_users_into_batches(user_to_description, batch_size):
    """
    Divide users and their descriptions into batches of a specified size.
    """
    description_batches = []
    user_id_batches = []
    users_with_descriptions = [
        user
        for user, description in user_to_description.items()
        if (isinstance(description, str) and description.strip())
        or (isinstance(description, list) and description and description[0].strip())
    ]

    for i in range(0, len(users_with_descriptions), batch_size):
        batch_users = users_with_descriptions[i : i + batch_size]
        user_id_batches.append(batch_users)
        description_batches.append(
            [
                user_to_description[user]
                if isinstance(user_to_description[user], str)
                else user_to_description[user][0]
                for user in batch_users
            ]
        )

    return user_id_batches, description_batches
