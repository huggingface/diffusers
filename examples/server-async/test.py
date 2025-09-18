import os
import time
import urllib.parse

import requests


SERVER_URL = "http://localhost:8500/api/diffusers/inference"
BASE_URL = "http://localhost:8500"
DOWNLOAD_FOLDER = "generated_images"
WAIT_BEFORE_DOWNLOAD = 2  # seconds

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


def save_from_url(url: str) -> str:
    """Download the given URL (relative or absolute) and save it locally."""
    if url.startswith("/"):
        direct = BASE_URL.rstrip("/") + url
    else:
        direct = url
    resp = requests.get(direct, timeout=60)
    resp.raise_for_status()
    filename = os.path.basename(urllib.parse.urlparse(direct).path) or f"img_{int(time.time())}.png"
    path = os.path.join(DOWNLOAD_FOLDER, filename)
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def main():
    payload = {
        "prompt": "The T-800 Terminator Robot Returning From The Future, Anime Style",
        "num_inference_steps": 30,
        "num_images_per_prompt": 1,
    }

    print("Sending request...")
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=480)
        r.raise_for_status()
    except Exception as e:
        print(f"Request failed: {e}")
        return

    body = r.json().get("response", [])
    # Normalize to a list
    urls = body if isinstance(body, list) else [body] if body else []
    if not urls:
        print("No URLs found in the response. Check the server output.")
        return

    print(f"Received {len(urls)} URL(s). Waiting {WAIT_BEFORE_DOWNLOAD}s before downloading...")
    time.sleep(WAIT_BEFORE_DOWNLOAD)

    for u in urls:
        try:
            path = save_from_url(u)
            print(f"Image saved to: {path}")
        except Exception as e:
            print(f"Error downloading {u}: {e}")


if __name__ == "__main__":
    main()
