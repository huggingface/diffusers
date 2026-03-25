import argparse
import os

from PIL import Image, ImageChops, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a side-by-side image comparison with diff.")
    parser.add_argument("--before", required=True, help="Path to the before image.")
    parser.add_argument("--after", required=True, help="Path to the after image.")
    parser.add_argument("--output", default="compare_before_after_diff.png", help="Output image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    before = Image.open(args.before).convert("RGB")
    after = Image.open(args.after).convert("RGB")
    diff = ImageChops.difference(before, after)

    width = before.width + after.width + diff.width
    height = max(before.height, after.height, diff.height) + 40

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(before, (0, 40))
    canvas.paste(after, (before.width, 40))
    canvas.paste(diff, (before.width + after.width, 40))

    draw = ImageDraw.Draw(canvas)
    labels = ["Before", "After", "Diff"]
    offsets = [0, before.width, before.width + after.width]

    for label, x_offset in zip(labels, offsets):
        draw.text((x_offset + 10, 10), label, fill=(0, 0, 0))

    canvas.save(args.output)
    print(f"Saved comparison to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
