import json
import yaml
import argparse
from src.llm import Embedder, OpenAIEmbedder, GigaChatEmbedder, E5Embedder, KaLMEmbedder, E5SmallEmbedder
from src.profile_generator.profile_generator import ProfileGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings and descriptions")
    parser.add_argument(
        "--embedder", type=str, required=True, help="Name of the embedder"
    )
    parser.add_argument(
        "--descriptions-path",
        type=str,
        required=True,
        help="Path to the json with descriptions",
    )
    parser.add_argument(
        "--embeddings-path", type=str, required=True, help="Path to save embeddings"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    match args.embedder:
        case "openai":
            with open("token.yaml") as f:
                token = yaml.safe_load(f)["openai"]
            embedder: Embedder = OpenAIEmbedder(token=token)
        case "gigachat":
            with open("token.yaml") as f:
                token = yaml.safe_load(f)["gigachat"]
            embedder: Embedder = GigaChatEmbedder(token=token)
        case "e5":
            embedder: Embedder = E5Embedder("intfloat/multilingual-e5-large")
        case "e5-small":
            embedder: Embedder = E5SmallEmbedder()
        case "kalm":
            embedder: Embedder = KaLMEmbedder()
        case _:
            raise ValueError(f"Unsupported embedder: {args.embedder}")

    # Read descriptions
    with open(args.descriptions_path, "r", encoding="utf-8") as f:
        user2descripton = json.load(f)

    embeddings = ProfileGenerator.generate_embeddings(user2descripton, embedder)
    # Save embeddings to file
    with open(args.embeddings_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)


if __name__ == "__main__":
    main()
