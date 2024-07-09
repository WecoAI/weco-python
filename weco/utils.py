import re
from typing import Dict, Optional

def extract_image_info(data_url: str) -> Optional[Dict[str, str]]:
    """
    Extract the image information from the encoded data or URL provided.

    Args:
        data_url (str): Encoded data or URL

    Returns:
        Optional[Dict[str, str]]: Image information
    """
    pattern = r'data:(?P<media_type>[\w/]+);(?P<source_type>\w+),(?P<encoding>.*)'
    match = re.match(pattern, data_url)
    if match:
        return match.groupdict()
    return None
