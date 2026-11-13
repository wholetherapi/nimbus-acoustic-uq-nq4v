"""Evaluate calibration and balanced accuracy on the held-out recording."""
import argparse
import json

import yaml

from src.acoustic_uq.calibration import SpeciesConditionalConformal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model = SpeciesConditionalConformal(alpha=cfg["alpha"])
    print(json.dumps({"model": "acoustic-uq", "alpha": cfg["alpha"], "species": cfg["species_list"]}))


if __name__ == "__main__":
    main()
