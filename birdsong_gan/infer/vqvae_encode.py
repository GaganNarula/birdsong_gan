import os
import json
import tqdm
import numpy as np
import torch
from datasets import load_from_disk, Dataset
from birdsong_gan.models.vqvae import VQVAEModel


def preprocess_example(x, device: str = "cuda"):
    x = np.array(x["spectrogram"])
    x = np.log1p(x)
    return torch.from_numpy(x).to(torch.float32).to(device)


def main(args):

    # load model
    print("Loading model...")
    model = VQVAEModel.from_pretrained(
        model_dir=args.model_dir, checkpoint_path=args.checkpoint_path
    )

    print("Model loaded.")

    print("Loading dataset...")
    # load data
    dataset = load_from_disk(args.dataset_path)

    print("Encoding data...")
    # preprocess and encode data
    encoded_data = {"codes": [], "bird_name": [], "days_post_hatch": []}
    for idx, example in enumerate(tqdm.tqdm(dataset)):
        x = preprocess_example(example)

        codes = model.infer_latent_codes_for_spectrogram(x)

        codes = codes.cpu().detach().numpy()
        encoded_data[idx] = codes

    # create Dataset object
    encoded_data = Dataset.from_dict(encoded_data)

    # save encoded data
    encoded_data.save_to_disk(args.output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode dataset with VQ-VAE model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset to be encoded.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory.",
    )

    args = parser.parse_args()
    main(args)
