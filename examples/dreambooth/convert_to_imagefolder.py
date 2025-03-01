import argparse
import json
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="Path to folder with image-text pairs.",
)
parser.add_argument("--caption_column", type=str, default="prompt", help="Name of caption column.")
args = parser.parse_args()

path = pathlib.Path(args.path)
if not path.exists():
    raise RuntimeError(f"`--path` '{args.path}' does not exist.")

all_files = list(path.glob("*"))
captions = list(path.glob("*.txt"))
images = set(all_files) - set(captions)
images = {image.stem: image for image in images}
caption_image = {caption: images.get(caption.stem) for caption in captions if images.get(caption.stem)}

metadata = path.joinpath("metadata.jsonl")

with metadata.open("w", encoding="utf-8") as f:
    for caption, image in caption_image.items():
        caption_text = caption.read_text(encoding="utf-8")
        json.dump({"file_name": image.name, args.caption_column: caption_text}, f)
        f.write("\n")
