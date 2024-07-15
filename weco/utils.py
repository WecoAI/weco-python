import base64
import os
import random
import re
import string
from io import BytesIO
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image


def is_local_image(maybe_local_image: str) -> bool:
    """
    Check if the file is a local image.

    Parameters
    ----------
    maybe_local_image : str
        The file path.

    Returns
    -------
    bool
        True if the file is a local image, False otherwise.
    """
    if not os.path.exists(maybe_local_image):  # Check if the file exists
        return False

    try:  # Check if the file is an image
        Image.open(maybe_local_image)
    except IOError:
        return False

    return True


def is_base64_image(maybe_base64: str) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    Check if the image is a base64 encoded image and if so, extract the image information from the encoded data or URL provided.

    Parameters
    ----------
    data : str
        The image data or URL.

    Returns
    -------
    Tuple[bool, Optional[Dict[str, str]]]
    """
    pattern = r"data:(?P<media_type>[\w/]+);(?P<source_type>\w+),(?P<encoding>.*)"
    match = re.match(pattern, maybe_base64)
    if match:
        return True, match.groupdict()

    return False, None


def is_public_url_image(maybe_url_image: str) -> bool:
    """
    Check if the string is a publicly accessible URL

    Parameters
    ----------
    maybe_url_image : str
        The URL to check.

    Returns
    -------
    bool
        True if the URL is publicly accessible, False otherwise.
    """
    try:
        # Check if it is a valid URL
        if not urlparse(maybe_url_image).scheme:
            return False

        # Check if the URL is publicly accessible
        response = requests.head(maybe_url_image)
        if response.status_code != 200:
            return False

        # Check if the URL is an image
        content_type = response.headers.get("content-type")
        if not content_type:
            return False
        if not content_type.startswith("image"):
            return False
    except Exception:
        return False

    return True


def get_image_size(image: str, source: str) -> float:
    """
    Get the size of the image in MB.

    Parameters
    ----------
    image : str
        The image data or URL.

    source : str
        The source of the image. It can be 'base64', 'url', or 'local'.

    Returns
    -------
    float
        The size of the image in MB.

    Raises
    ------
    ValueError
        If the image is not a valid image.
    """
    if source == "base64":
        _, base64_info = is_base64_image(maybe_base64=image)
        img_data = base64.b64decode(base64_info["encoding"])
    elif source == "url":
        response = requests.get(image)
        response.raise_for_status()
        img_data = response.content
    elif source == "local":
        with open(image, "rb") as f:
            img_data = f.read()
    else:
        raise ValueError("Invalid image input")

    img = Image.open(BytesIO(img_data))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format=img.format)
    return img_byte_arr.tell() / 1000000  # MB


def preprocess_image(image: Image, file_type: str) -> Tuple:
    """
    Preprocess the image by converting it to RGB if it has an alpha channel.

    Parameters
    ----------
    image : Image
        The image to preprocess.
    file_type : str
        The file type of the image.

    Returns
    -------
    Image
        The preprocessed image.
    file_type : str
        The file type of the image.
    """
    # Do not rescale or resize. Only do this if latency becomes an issue.
    # Remove the alpha channel for PNG and WEBP images if it exists.
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGB")

    # If the image file type is JPG, convert to JPEG for PIL compatibility.
    if file_type == "jpg":
        file_type = "jpeg"
    return image, file_type


def generate_random_base16_code(length: int = 5):
    """
    Generate a random base16 code.

    Parameters
    ----------
    length : int
        The length of the code.

    Returns
    -------
    str
        The random base16 code.
    """
    return "".join(random.choices(string.hexdigits, k=length))
