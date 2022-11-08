import os
import tempfile
import textwrap

import rclonepy as rclone


REMOTE_PATH = "azpublic:$web/diffusers"
INDEX_URL = "https://characterpublic.z21.web.core.windows.net/diffusers/"


def main():
    index = os.path.join(REMOTE_PATH, "index.html")
    print(f"Updating {index} ... ", flush=True, end="")
    with tempfile.NamedTemporaryFile("w+", suffix=".html") as h:
        wheels = "\n".join(
            [
                f'<a href="/diffusers/{wheel}">{wheel}</a><br />'
                for wheel in rclone.lsf(REMOTE_PATH)
                if wheel.endswith(".whl")
            ]
        )
        print(
            textwrap.dedent(
                f"""
                <!DOCTYPE html>
                <html>
                    <body>
                        {wheels}
                    </body>
                </html>
                """
            ).strip(),
            file=h,
            flush=True,
        )
        rclone.copyto(h.name, index)
    print("done")
    print(f"Updated package index.html: {INDEX_URL}")


if __name__ == "__main__":
    main()
