import json
from abc import abstractmethod, ABC
from typing import Literal, List, Any, Dict
from src.llm import LLM


class UserProfileGenerator(ABC):
    def __init__(self, dataset_name, dataset_type: Literal["long", "short"]):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type

    @abstractmethod
    def generate_description(
        self, user_interactions: dict, llm: LLM, max_output_tokens: int
    ):
        pass

    @abstractmethod
    def check_number_of_splits(
        self, user_interactions: list[dict], llm: LLM, max_output_tokens: int
    ):
        pass

    @abstractmethod
    def generate_description_batch(
        self, users_interactions: list[dict], llm: LLM, max_output_tokens: int
    ):
        pass


class AggAfterPipeline(UserProfileGenerator):
    def __init__(self, dataset_name, dataset_type: Literal["long", "short"]):
        super(AggAfterPipeline, self).__init__(dataset_name, dataset_type)
        from src.prompts.sequential_agg_after import PROMPTS_DICT

        self.prompt_create = PROMPTS_DICT[dataset_type][dataset_name]["create_profile"]
        self.prompt_aggregate = PROMPTS_DICT[dataset_type][dataset_name][
            "aggregate_profiles"
        ]

    def generate_description(self, user_interactions, llm: LLM, max_output_tokens: int):
        def _generate_profile(cur_user_interactions):
            messages = self.create_messages_user_interactions(
                cur_user_interactions, llm
            )
            if (
                llm.get_tokens_length(messages) + max_output_tokens
                < llm.get_max_context_length()
            ):
                return [
                    llm.get_description(messages, max_output_tokens=max_output_tokens)
                ]
            if len(cur_user_interactions) == 1:
                return []
            split_i = len(cur_user_interactions) // 2
            profiles_left = _generate_profile(cur_user_interactions[:split_i])
            profiles_right = _generate_profile(cur_user_interactions[split_i:])
            profiles_left.extend(profiles_right)
            return profiles_left

        profiles = _generate_profile(user_interactions)
        if len(profiles) == 1:
            return profiles[0]
        messages = self.create_messages_user_profiles(profiles, llm)
        return llm.get_description(messages, max_output_tokens=max_output_tokens)

    def generate_description_batch(
        self, users_interactions: list[list[dict]], llm: LLM, max_output_tokens: int
    ) -> List[str]:
        messages: List[List[Any]] = [
            self.create_messages_user_interactions(user_interactions, llm)
            for user_interactions in users_interactions
        ]

        return llm.get_description(messages, max_output_tokens=max_output_tokens)

    def check_number_of_splits(
        self, user_interactions: list[dict], llm: LLM, max_output_tokens: int
    ):
        """
        Compute the number of splits required to generate a description of the
        given user interactions with the given language model and maximum output
        tokens.

        Args:
            user_interactions (list[dict]): The user interactions.
            llm (LLM): The language model.
            max_output_tokens (int): The maximum number of output tokens.

        Returns:
            int: The number of splits required.
        """

        def _check_number_of_splits(cur_user_interactions: list[dict]):
            messages = self.create_messages_user_interactions(
                cur_user_interactions, llm
            )
            if (
                llm.get_tokens_length(messages) + max_output_tokens
                < llm.get_max_context_length()
            ):
                return 1
            if len(cur_user_interactions) == 1:
                return 0
            split_i = len(cur_user_interactions) // 2
            return _check_number_of_splits(
                cur_user_interactions[:split_i]
            ) + _check_number_of_splits(cur_user_interactions[split_i:])

        return _check_number_of_splits(user_interactions)

    def create_messages_user_interactions(
        self, user_interactions, llm: LLM
    ) -> List[Any]:
        user_interactions__str = json.dumps(user_interactions, ensure_ascii=False)
        prompt = self.prompt_create.replace("{user_history}", user_interactions__str)
        return llm.prepare_input(prompt)

    def create_messages_user_profiles(self, user_profiles, llm: LLM) -> List[Any]:
        user_profiles__str = json.dumps(user_profiles)
        prompt = self.prompt_aggregate.replace("{user_profiles}", user_profiles__str)
        return llm.prepare_input(prompt)


class AggWithPipeline(UserProfileGenerator):
    def __init__(self, dataset_name, dataset_type: Literal["long", "short"]):
        super(AggWithPipeline, self).__init__(dataset_name, dataset_type)
        from src.prompts.sequential_with_agg import PROMPTS_DICT

        self.prompt_create = PROMPTS_DICT[dataset_type][dataset_name][
            "create_first_profile"
        ]
        self.update_profile = PROMPTS_DICT[dataset_type][dataset_name]["update_profile"]

    def generate_description(
        self, user_interactions, llm: LLM, max_output_tokens: int
    ) -> str:
        """
        Generate a user profile description by processing user interactions sequentially.
        The first step creates an initial profile, and subsequent steps update the profile
        with additional interactions.

        Args:
            user_interactions (list[dict]): The list of user interactions.
            llm (LLM): The language model.
            max_output_tokens (int): The maximum number of output tokens.

        Returns:
            str: The final user profile description.
        """

        def _generate_profile(cur_user_interactions, current_profile: str = "") -> str:
            if not current_profile:
                messages = self.create_messages_first_step(cur_user_interactions, llm)
            else:
                messages = self.create_messages_other_steps(
                    cur_user_interactions, current_profile, llm
                )

            if (
                llm.get_tokens_length(messages) + max_output_tokens
                < llm.get_max_context_length()
            ):
                return llm.get_description(
                    messages, max_output_tokens=max_output_tokens
                )

            if len(cur_user_interactions) == 1:
                return current_profile

            split_i = len(cur_user_interactions) // 2
            profile_left = _generate_profile(
                cur_user_interactions[:split_i], current_profile
            )
            profile_final = _generate_profile(
                cur_user_interactions[split_i:], profile_left
            )
            return profile_final

        profile = _generate_profile(user_interactions, "")
        return profile

    def generate_description_batch(
        self, users_interactions: list[list[dict]], llm: LLM, max_output_tokens: int
    ) -> List[str]:
        """
        Generate user profile descriptions for multiple users in batch mode.
        Each user's profile is generated independently using the sequential pipeline.

        Args:
            users_interactions (list[list[dict]]): List of user interactions for multiple users.
            llm (LLM): The language model.
            max_output_tokens (int): The maximum number of output tokens.

        Returns:
            List[str]: List of user profile descriptions.
        """
        messages: List[List[Any]] = [
            self.create_messages_first_step(user_interactions, llm)
            for user_interactions in users_interactions
        ]

        return llm.get_description(messages, max_output_tokens=max_output_tokens)

    def check_number_of_splits(
        self, user_interactions: list[dict], llm: LLM, max_output_tokens: int
    ):
        """
        Compute the number of splits required to generate a description of the
        given user interactions with the given language model and maximum output
        tokens.

        Args:
            user_interactions (list[dict]): The user interactions.
            llm (LLM): The language model.
            max_output_tokens (int): The maximum number of output tokens.

        Returns:
            int: The number of splits required.
        """

        def _check_number_of_splits(
            cur_user_interactions: list[dict], current_profile: str = ""
        ) -> int:
            if not current_profile:
                messages = self.create_messages_first_step(cur_user_interactions, llm)
            else:
                messages = self.create_messages_other_steps(
                    cur_user_interactions, current_profile, llm
                )

            if (
                llm.get_tokens_length(messages) + max_output_tokens
                < llm.get_max_context_length()
            ):
                return 1
            if len(cur_user_interactions) == 1:
                return 0
            split_i = len(cur_user_interactions) // 2
            return _check_number_of_splits(
                cur_user_interactions[:split_i], current_profile
            ) + _check_number_of_splits(
                cur_user_interactions[split_i:], "dummy_profile"
            )

        return _check_number_of_splits(user_interactions)

    def create_messages_first_step(
        self, user_interactions: list[dict], llm: LLM
    ) -> List[Any]:
        """
        Create messages for the first step of profile generation.

        Args:
            user_interactions (list[dict]): The user interactions.
            llm (LLM): The language model.

        Returns:
            List[Any]: Prepared messages for the LLM.
        """
        user_interactions__str = json.dumps(user_interactions, ensure_ascii=False)
        prompt = self.prompt_create.replace("{user_history}", user_interactions__str)
        return llm.prepare_input(prompt)

    def create_messages_other_steps(
        self, user_interactions: list[dict], user_profile: str, llm: LLM
    ) -> List[Any]:
        """
        Create messages for updating an existing profile with new interactions.

        Args:
            user_interactions (list[dict]): The user interactions.
            user_profile (str): The current user profile.
            llm (LLM): The language model.

        Returns:
            List[Any]: Prepared messages for the LLM.
        """
        user_interactions__str = json.dumps(user_interactions, ensure_ascii=False)

        if isinstance(user_profile, list):
            user_profile = user_profile[0]
        prompt = self.update_profile.replace("{user_profile}", user_profile).replace(
            "{user_history}", user_interactions__str
        )
        return llm.prepare_input(prompt)


class SeveralProfilesPipeline(UserProfileGenerator):
    def __init__(self, dataset_name, dataset_type: Literal["long", "short"]):
        super(SeveralProfilesPipeline, self).__init__(dataset_name, dataset_type)
        from src.prompts.sequential_several import PROMPTS_DICT

        self.prompts = PROMPTS_DICT[dataset_type][dataset_name]
        self.prompts_types = [key for key in self.prompts.keys() if key != "aggregate"]

    def generate_description(
        self, user_interactions: dict, llm: LLM, max_output_tokens: int
    ) -> Dict[str, str]:
        def _generate_profile(cur_user_interactions, prompt_type):
            messages = self.create_messages_user_interactions(
                cur_user_interactions, prompt_type, llm
            )
            if (
                llm.get_tokens_length(messages) + max_output_tokens
                < llm.get_max_context_length()
            ):
                return [
                    llm.get_description(messages, max_output_tokens=max_output_tokens)
                ]
            if len(cur_user_interactions) == 1:
                return []
            split_i = len(cur_user_interactions) // 2
            profiles_left = _generate_profile(
                cur_user_interactions[:split_i], prompt_type
            )
            profiles_right = _generate_profile(
                cur_user_interactions[split_i:], prompt_type
            )
            profiles_left.extend(profiles_right)
            return profiles_left

        output = {}
        for prompt_type in self.prompts_types:
            profiles = _generate_profile(user_interactions, prompt_type)

            if len(profiles) == 1:
                output[prompt_type] = profiles[0][0]
            else:
                messages = self.create_messages_user_profiles(
                    [profile[0] for profile in profiles], llm
                )
                output[prompt_type] = llm.get_description(
                    messages, max_output_tokens=max_output_tokens
                )

        return output

    def check_number_of_splits(
        self, user_interactions: list[dict], llm: LLM, max_output_tokens: int
    ):
        def _check_number_of_splits(cur_user_interactions, prompt_type):
            messages = self.create_messages_user_interactions(
                cur_user_interactions, prompt_type, llm
            )
            if (
                llm.get_tokens_length(messages) + max_output_tokens
                < llm.get_max_context_length()
            ):
                return 1
            if len(cur_user_interactions) == 1:
                return 0
            split_i = len(cur_user_interactions) // 2
            return _check_number_of_splits(
                cur_user_interactions[:split_i], prompt_type
            ) + _check_number_of_splits(cur_user_interactions[split_i:], prompt_type)

        sum_number_of_splits = sum(
            [
                _check_number_of_splits(user_interactions, prompt_type)
                for prompt_type in self.prompts_types
            ]
        )
        if sum_number_of_splits <= len(self.prompts_types):
            return 1
        return sum_number_of_splits

    def generate_description_batch(
        self, users_interactions: list[dict], llm: LLM, max_output_tokens: int
    ) -> Dict[str, List[str]]:
        output = {prompt_type: [] for prompt_type in self.prompts_types}

        for prompt_type in self.prompts_types:
            messages: List[List[Any]] = [
                self.create_messages_user_interactions(
                    user_interactions, prompt_type, llm
                )
                for user_interactions in users_interactions
            ]

            descriptions = llm.get_description(
                messages, max_output_tokens=max_output_tokens
            )
            output[prompt_type].extend(descriptions)

        return output

    def create_messages_user_interactions(
        self, user_interactions, prompt_type, llm: LLM
    ) -> List[Any]:
        user_interactions__str = json.dumps(user_interactions, ensure_ascii=False)
        prompt = self.prompts[prompt_type].replace(
            "{user_history}", user_interactions__str
        )
        return llm.prepare_input(prompt)

    def create_messages_user_profiles(self, user_profiles, llm: LLM) -> List[Any]:
        user_profiles__str = json.dumps(user_profiles)
        prompt = self.prompts["aggregate"].replace(
            "{user_profiles}", user_profiles__str
        )
        return llm.prepare_input(prompt)
