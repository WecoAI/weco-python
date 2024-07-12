import os
import re
import requests
from urllib.parse import urlparse
from typing import Dict, Optional, Tuple
from io import BytesIO
import base64
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
    # Check if it is a valid URL
    if not urlparse(maybe_url_image).scheme:
        return False

    # Check if the URL is publicly accessible
    response = requests.head(maybe_url_image)
    if response.status_code != 200:
        return False

    # Check if the URL is an image
    content_type = response.headers.get('content-type')
    if not content_type:
        return False
    if not content_type.startswith('image'):
        return False

    return True


def get_image_size(image: str) -> float:
    """
    Get the size of the image in MB.

    Args:
        image (str): URL, local path, or base64 encoding of the image

    Returns:
        float: Size of the image in MB.

    Raises:
        ValueError: If the image is not a valid input.
    """
    is_base64, image_info = is_base64_image(maybe_base64=image)

    if is_base64:
        img_data = base64.b64decode(image_info['encoding'])
    elif is_public_url_image(maybe_url_image=image):
        response = requests.get(image)
        response.raise_for_status()
        img_data = response.content
    elif is_local_image(maybe_local_image=image):
        with open(image, 'rb') as f:
            img_data = f.read()
    else:
        raise ValueError("Invalid image input")

    img = Image.open(BytesIO(img_data))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format=img.format)
    return img_byte_arr.tell() / 1000000  # MB


def preprocess_image(image: Image) -> Image:
    """
    Preprocess the image by converting it to RGB if it has an alpha channel.

    Parameters
    ----------
    image : Image
        The image to preprocess.

    Returns
    -------
    Image
        The preprocessed image
    """
    # Do not rescale or resize. Only do this if latency becomes an issue.
    # Remove the alpha channel for PNG and WEBP images if it exists.
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert('RGB')
    return image
