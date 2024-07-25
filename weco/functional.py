from typing import Any, Dict, List, Optional

from .client import WecoAI

# TODO: Implement the closing stuff for the client
def build(task_description: str, api_key: str = None) -> tuple[str, str]:
    """Builds a specialized function synchronously given a task description.

    Parameters
    ----------
    task_description : str
        A description of the task for which the function is being built.
    api_key : str
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - WECO_API_KEY.

    Returns
    -------
    tuple[str, str]
        A tuple containing the name and description of the function.
    """
    client = WecoAI(api_key=api_key)
    response = client.build(task_description=task_description)
    return response


async def abuild(task_description: str, api_key: str = None) -> tuple[str, str]:
    """Builds a specialized function asynchronously given a task description.

    Parameters
    ----------
    task_description : str
        A description of the task for which the function is being built.
    api_key : str
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - WECO_API_KEY.

    Returns
    -------
    tuple[str, str]
        A tuple containing the name and description of the function.
    """
    client = WecoAI(api_key=api_key)
    response = await client.abuild(task_description=task_description)
    return response


def query(
    fn_name: str, text_input: Optional[str] = "", images_input: Optional[List[str]] = [], api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Queries a function synchronously with the given function ID and input.

    Parameters
    ----------
    fn_name : str
        The name of the function to query.
    text_input : str, optional
        The text input to the function.
    images_input : List[str], optional
        A list of image URLs or base64 encoded images to be used as input to the function.
    api_key : str
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - WECO_API_KEY.

    Returns
    -------
    dict
        A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
        and the latency in milliseconds.
    """
    client = WecoAI(api_key=api_key)
    response = client.query(fn_name=fn_name, text_input=text_input, images_input=images_input)
    return response


async def aquery(
    fn_name: str, text_input: Optional[str] = "", images_input: Optional[List[str]] = [], api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Queries a function asynchronously with the given function ID and input.

    Parameters
    ----------
    fn_name : str
        The name of the function to query.
    text_input : str, optional
        The text input to the function.
    images_input : List[str], optional
        A list of image URLs to be used as input to the function.
    api_key : str
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - WECO_API_KEY.

    Returns
    -------
    dict
        A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
        and the latency in milliseconds.
    """
    client = WecoAI(api_key=api_key)
    response = await client.aquery(fn_name=fn_name, text_input=text_input, images_input=images_input)
    return response


def batch_query(
    fn_names: str | List[str], batch_inputs: List[Dict[str, Any]], api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Synchronously queries multiple functions using asynchronous calls internally.

    This method uses the asynchronous queries to submit all queries concurrently
    and waits for all responses to be received before returning the results.

    Parameters
    ----------
    fn_name : str | List[str]
        The name of the function or a list of function names to query.
        Note that if a single function name is provided, it will be used for all queries.
        If a list of function names is provided, the length must match the number of queries.

    batch_inputs : List[str]
       A list of inputs for the functions to query. The input must be a dictionary containing the data to be processed. e.g.,
       when providing for a text input, the dictionary should be {"text_input": "input text"}, for an image input, the dictionary should be {"images_input": ["url1", "url2", ...]}
       and for a combination of text and image inputs, the dictionary should be {"text_input": "input text", "images_input": ["url1", "url2", ...]}.
       Note that the index of each input must correspond to the index of the function name when both inputs are lists.

    api_key : str, optional
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - WECO_API_KEY.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing the output of a function query,
        in the same order as the input queries.
    """
    client = WecoAI(api_key=api_key)
    responses = client.batch_query(fn_names=fn_names, batch_inputs=batch_inputs)
    return responses
