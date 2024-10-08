from typing import Any, Dict, List, Optional, Union

from .client import WecoAI


def build(task_description: str, multimodal: bool = False, api_key: Optional[str] = None) -> tuple[str, int, str]:
    """Build a specialized function for a task.

    Parameters
    ----------
    task_description : str
        The description of the task for which the function is being built.

    multimodal : bool, optional
        A flag to indicate if the function should be multimodal. Default is False.

    api_key : str, optional
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - `WECO_API_KEY`.

    Returns
    -------
    tuple[str, int, str]
        A tuple containing the function name, version number and description.
    """
    client = WecoAI(api_key=api_key)
    response = client.build(task_description=task_description, multimodal=multimodal)
    return response


async def abuild(task_description: str, multimodal: bool = False, api_key: Optional[str] = None) -> tuple[str, int, str]:
    """Build a specialized function for a task asynchronously.

    Parameters
    ----------
    task_description : str
        The description of the task for which the function is being built.

    multimodal : bool, optional
        A flag to indicate if the function should be multimodal. Default is False.

    api_key : str, optional
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - `WECO_API_KEY`.

    Returns
    -------
    tuple[str, int, str]
        A tuple containing the function name, version number and description.
    """
    client = WecoAI(api_key=api_key)
    response = await client.abuild(task_description=task_description, multimodal=multimodal)
    return response


def query(
    fn_name: str,
    version: Optional[Union[str, int]] = -1,
    version_number: Optional[int] = -1,
    text_input: Optional[str] = "",
    images_input: Optional[List[str]] = [],
    return_reasoning: Optional[bool] = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Queries a specific function with the input (text, images or both).

    Parameters
    ----------
    fn_name : str
        The name of the function to query.

    version : str | int, optional
        The version alias/number of the function to query. Default is -1 which results in the latest version being used.

    version_number : int, optional
        The version number of the function to query. Default is -1 which results in the latest version being used.

    text_input : str, optional
        The text input to the function.

    images_input : List[str], optional
        A list of image URLs or base64 encoded images to be used as input to the function.

    return_reasoning : bool, optional
        A flag to indicate if the reasoning should be returned. Default is False.

    api_key : str, optional
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - `WECO_API_KEY`.


    Returns
    -------
    dict
        A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
        and the latency in milliseconds.
    """
    client = WecoAI(api_key=api_key)
    response = client.query(
        fn_name=fn_name,
        version=version,
        version_number=version_number,
        text_input=text_input,
        images_input=images_input,
        return_reasoning=return_reasoning,
    )
    return response


async def aquery(
    fn_name: str,
    version: Optional[Union[str, int]] = -1,
    version_number: Optional[int] = -1,
    text_input: Optional[str] = "",
    images_input: Optional[List[str]] = [],
    return_reasoning: Optional[bool] = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Queries a specific function with the input (text, images or both) asynchronously.

    Parameters
    ----------
    fn_name : str
        The name of the function to query.

    version: str | int, optional
        The version alias/number of the function to query. Default is -1 which results in the latest version being used.

    version_number : int, optional
        The version number of the function to query. Default is -1 which results in the latest version being used.

    text_input : str, optional
        The text input to the function.

    images_input : List[str], optional
        A list of image URLs or base64 encoded images to be used as input to

    return_reasoning : bool, optional
        A flag to indicate if the reasoning should be returned. Default is False.

    api_key : str
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - `WECO_API_KEY`.

    Returns
    -------
    dict
        A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
    """
    client = WecoAI(api_key=api_key)
    response = await client.aquery(
        fn_name=fn_name,
        version=version,
        version_number=version_number,
        text_input=text_input,
        images_input=images_input,
        return_reasoning=return_reasoning,
    )
    return response


def batch_query(
    fn_name: str,
    batch_inputs: List[Dict[str, Any]],
    version: Optional[Union[str, int]] = -1,
    version_number: Optional[int] = -1,
    return_reasoning: Optional[bool] = False,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Batch queries a single function with multiple inputs.

    This method uses the asynchronous queries to submit a batch of queries concurrently
    and waits for all responses to be received before returning the results.
    Order of the responses corresponds to the order of the inputs.

    Parameters
    ----------
    fn_name : str
        The name of the function to query.

    batch_inputs : List[Dict[str, Any]]
        A list of dictionaries, each representing an input for the function. Each dictionary can contain:
        - "text_input": A string for text input.
        - "images_input": A list of image URLs or base64 encoded images.

    version : str | int, optional
        The version alias/number of the function to query. Default is -1 which results in the latest version being used.

    version_number : int, optional
        The version number of the function to query. If not provided, the latest version is used. Default is -1 for the same behavior.

    return_reasoning : bool, optional
        A flag to indicate if the reasoning should be returned. Default is False.

    api_key : str, optional
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - `WECO_API_KEY`.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing the result of a function query. Each dictionary includes:
        - The function's output.
        - The number of input tokens.
        - The number of output tokens.
        - The latency in milliseconds.
    """
    client = WecoAI(api_key=api_key)
    responses = client.batch_query(
        fn_name=fn_name,
        version=version,
        version_number=version_number,
        batch_inputs=batch_inputs,
        return_reasoning=return_reasoning,
    )
    return responses
