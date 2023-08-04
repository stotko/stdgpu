from __future__ import annotations

import difflib
import html
import pathlib


def main() -> None:
    doxygen_dir = pathlib.Path(__file__).parents[1] / "doxygen"

    print(f'Fixing HTML titles in "{doxygen_dir}"...')

    counter = 0
    for file in sorted((pathlib.Path(__file__).parents[1] / "doxygen").glob("*rst")):
        with file.open("r") as f:
            rst_content = f.read()

            rst_content_unescaped = html.unescape(rst_content)

            if rst_content_unescaped != rst_content:
                counter += 1
                diff = difflib.unified_diff(
                    rst_content.splitlines(keepends=True), rst_content_unescaped.splitlines(keepends=True)
                )

                print("".join(diff), end="")

        with file.open("w") as f:
            f.write(rst_content_unescaped)

    print(f'Fixing HTML titles in "{doxygen_dir}"... done. ({counter} files changed)')


if __name__ == "__main__":
    main()
