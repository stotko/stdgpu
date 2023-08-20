from __future__ import annotations

import skimage.color
import numpy as np


def contrast(rgb1, rgb2) -> float:
    xyz1 = skimage.color.rgb2xyz(np.array(rgb1) / 255)
    xyz2 = skimage.color.rgb2xyz(np.array(rgb2) / 255)

    l1 = max(xyz1[1], xyz2[1])
    l2 = min(xyz1[1], xyz2[1])

    return (l1 + 0.05) / (l2 + 0.05)


def main() -> None:
    # Sphinx book theme
    light_background = (255, 255, 255)
    light_text = (50, 50, 50)
    light_header = (100, 100, 100)
    light_separator = (201, 201, 201)

    # Modified Sphinx book theme (with original values if changed)
    dark_background = (18, 18, 18)
    dark_text = (206, 206, 206)
    dark_header = (166, 166, 166)
    dark_separator = (58, 58, 58)  # (192, 192, 192)

    print("Light:", f"{contrast(light_background, light_text):.3f}",
                    f"{contrast(light_background, light_header):.3f}",
                    f"{contrast(light_background, light_separator):.3f}",
                    f"{contrast(light_separator, light_text):.3f}",
                    f"{contrast(light_separator, light_header):.3f}")

    print("Dark: ", f"{contrast(dark_background, dark_text):.3f}",
                    f"{contrast(dark_background, dark_header):.3f}",
                    f"{contrast(dark_background, dark_separator):.3f}",
                    f"{contrast(dark_separator, dark_text):.3f}",
                    f"{contrast(dark_separator, dark_header):.3f}")


if __name__ == "__main__":
    main()
