from pathlib import Path
from PIL import Image

INPUT_DIR = Path("/medical_images/output")
OUTPUT_DIR = Path("/medical_images/output_cropped")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_smallest_size(image_paths: list[Path]) -> tuple[int, int]:
    min_width = None
    min_height = None

    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size

        if min_width is None or width < min_width:
            min_width = width
        if min_height is None or height < min_height:
            min_height = height

    if min_width is None or min_height is None:
        raise ValueError("No readable images found.")

    return min_width, min_height


def crop_image_to_center(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    left = (image.width - target_width) // 2
    top = (image.height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    return image.crop((left, top, right, bottom))


def main() -> None:
    image_paths = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    target_width, target_height = find_smallest_size(image_paths)
    print(f"Smallest size found: {target_width}x{target_height}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        with Image.open(path) as img:
            if img.width < target_width or img.height < target_height:
                print(f"[SKIP] {path.name}: image smaller than target size")
                continue

            cropped = crop_image_to_center(img, target_width, target_height)
            out_path = OUTPUT_DIR / path.name
            cropped.save(out_path)
            print(f"[OK] {path.name} -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()