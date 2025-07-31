import json
import os
import yaml
import argparse

from src.profile_generator.generation_pipelines import (
    AggAfterPipeline,
    AggWithPipeline,
    SeveralProfilesPipeline,
)
from src.llm import LLM, Gemma2, GigaChatLLM, OpenAILLM, Llama3
from src.profile_generator.profile_generator import (
    ProfileGenerator,
    SeveralProfileGenerator,
)

from src.datasets import (
    Dataset,
    KionDataset,
    BeautyDataset,
    ReesDataset,
    MovieLensDataset,
    AmazonM2Dataset,
)
from src.utils import read_json


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings and descriptions")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=False,
        help="Type of the dataset: train or val",
        default="train",
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument("--llm", type=str, required=True, help="Name of the LLM")
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        required=False,
        default=None,
        help="Max output tokens",
    )
    parser.add_argument(
        "--prompts-type",
        required=False,
        type=str,
        default="long",
        help="Prompts type: long or short",
    )
    parser.add_argument(
        "--long-gen-strategy",
        type=str,
        required=False,
        default="agg_after",
        help="Profile aggregation strategy. agg_after - combine profiles for samples after calculating all samples. agg_with - combine profiles for samples sequentially",
    )
    parser.add_argument(
        "--descriptions-path",
        type=str,
        required=True,
        help="Path to save files with descriptions",
    )
    parser.add_argument(
        "--user-ids-path",
        type=str,
        required=False,
        default=None,
        help="Path to user ids",
    )
    return parser.parse_args()


def _main(args):
    match args.dataset:
        case "kion":
            dataset: Dataset = KionDataset()
            interaction_field = "watched %"
        case "beauty":
            dataset: Dataset = BeautyDataset(args.dataset_type)
            interaction_field = "ratings"
        case "rees46":
            dataset: Dataset = ReesDataset()
            interaction_field = "event_type"
        case "ml-20m":
            dataset: Dataset = MovieLensDataset(args.dataset_type)
            interaction_field = "ratings"
        case "amazon-m2":
            dataset = AmazonM2Dataset()
            interaction_field = "interactions"
        case _:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset.load_data(args.dataset_path)

    llm_kwargs = {}
    match args.llm:
        case "openai":
            with open("token.yaml") as f:
                token = yaml.safe_load(f)["openai"]
            llm_kwargs["token"] = token
            llm_kwargs["model"] = "gpt-4o-mini"
        case "gigachat":
            with open("token.yaml") as f:
                token = yaml.safe_load(f)["gigachat"]
            llm_kwargs["token"] = token
        case "gemma2":
            with open("token.yaml") as f:
                token = yaml.safe_load(f)["huggingface"]
            llm_kwargs["token"] = token
        case "llama3":
            with open("token.yaml") as f:
                token = yaml.safe_load(f)["huggingface"]
            llm_kwargs["token"] = token
        case _:
            raise ValueError(f"Unsupported LLM: {args.llm}")

    if args.max_output_tokens is not None:
        llm_kwargs["max_tokens"] = args.max_output_tokens

    match args.llm:
        case "openai":
            llm: LLM = OpenAILLM(**llm_kwargs)
        case "gigachat":
            llm = GigaChatLLM(**llm_kwargs)
        case "gemma2":
            llm = Gemma2(**llm_kwargs)
        case "llama3":
            llm = Llama3(**llm_kwargs)

    match args.long_gen_strategy:
        case "agg_after":
            gen_pipeline = AggAfterPipeline(args.dataset, args.prompts_type)
            profile_generator = ProfileGenerator(gen_pipeline)
        case "agg_with":
            gen_pipeline = AggWithPipeline(args.dataset, args.prompts_type)
            profile_generator = ProfileGenerator(gen_pipeline)
        case "several_profiles":
            os.makedirs(os.path.join(args.descriptions_path, "type1"), exist_ok=True)
            os.makedirs(os.path.join(args.descriptions_path, "type2"), exist_ok=True)
            os.makedirs(os.path.join(args.descriptions_path, "type3"), exist_ok=True)

            gen_pipeline = SeveralProfilesPipeline(args.dataset, args.prompts_type)
            profile_generator = SeveralProfileGenerator(gen_pipeline)
        case _:
            raise ValueError(
                f"Unsupported aggregation strategy: {args.long_gen_strategy}"
            )

    if args.user_ids_path is None:
        user_ids = None
    else:
        user_ids = read_json(args.user_ids_path)

    os.makedirs(args.descriptions_path, exist_ok=True)
    return dataset, llm, profile_generator, user_ids, interaction_field


def run(args):
    dataset, llm, profile_generator, user_ids, interaction_field = _main(args)

    descriptions = profile_generator.generate_descriptions(
        dataset,
        llm,
        max_output_tokens=int(args.max_output_tokens),
        user_ids=user_ids,
        descriptions_path=args.descriptions_path,
        interaction_field=interaction_field,
    )

    if args.long_gen_strategy == "several_profiles":
        descriptions_types = {"type1": {}, "type2": {}, "type3": {}}

        for user_id, descriptions_list in descriptions.items():
            for i, prompt_type in enumerate(list(descriptions_list.keys()), 1):
                descriptions_types[f"type{i}"][user_id] = descriptions_list[
                    prompt_type
                ][0]

        for i in range(1, 4):
            with open(
                os.path.join(args.descriptions_path, f"descritions_type{i}_all.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(descriptions_types[f"type{i}"], f, ensure_ascii=False)
    else:
        with open(
            os.path.join(args.descriptions_path, "descriptions_all.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(descriptions, f, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    run(args)
