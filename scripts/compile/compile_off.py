import os.path

import requests
import yaml

from aimnet.train.pt2jpt import jitcompile


def compile_from_config(config):
    for job_name, job_config in config.items():
        print(f"Compiling {job_name}.")
        models = job_config.pop("models")
        job_config = _maybe_download(job_config)
        for task_config in models:
            task_config = _maybe_download(task_config)
            config = {**job_config, **task_config}
            print(f"{config['pt']} -> {config['jpt']}")
            jitcompile.callback(**config)  # type: ignore


def _maybe_download(d: dict[str, str]) -> dict[str, str]:
    for key, value in d.items():
        if value.startswith("https:"):
            filename = value.split("/")[-1]
            if not os.path.exists(filename):
                print(f"Downloading {filename}.")
                with open(filename, "wb") as file:
                    response = requests.get(value, timeout=20)
                    file.write(response.content)
            value = filename
        d[key] = value
    return d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch compile PyTorch models to TorchScript.")
    parser.add_argument("config", type=str, help="Path to the input YAML config file.")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file.read(), Loader=yaml.SafeLoader)

    compile_from_config(config)
