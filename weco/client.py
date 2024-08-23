import asyncio
import base64
import os
import warnings
from io import BytesIO
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

import httpx
import requests
from httpx import HTTPStatusError
from PIL import Image

from .constants import MAX_IMAGE_SIZE_MB, MAX_IMAGE_UPLOADS, MAX_TEXT_LENGTH, SUPPORTED_IMAGE_EXTENSIONS
from .utils import (
    generate_random_base16_code,
    get_image_size,
    is_base64_image,
    is_local_image,
    is_public_url_image,
    preprocess_image,
)


class WecoAI:
    """A client for the WecoAI function builder API that allows users to build and query specialized functions built by LLMs.
    The user must simply provide a task description to build a function, and then query the function with an input to get the result they need.
    Our client supports both synchronous and asynchronous request paradigms and uses HTTP/2 for faster communication with the API.
    Support for multimodality is included.

    Attributes
    ----------
    api_key : str
        The API key used for authentication.

    timeout : float
        The timeout for the HTTP requests in seconds. Default is 120.0.

    http2 : bool
        Whether to use HTTP/2 protocol for the HTTP requests. Default is True.
    """

    def __init__(self, api_key: Union[str, None] = None, timeout: float = 120.0, http2: bool = True) -> None:
        """Initializes the WecoAI client with the provided API key and base URL.

        Parameters
        ----------
        api_key : str, optional
            The API key used for authentication. If not provided, the client will attempt to read it from the environment variable - WECO_API_KEY.

        timeout : float, optional
            The timeout for the HTTP requests in seconds (default is 120.0).

        http2 : bool, optional
            Whether to use HTTP/2 protocol for the HTTP requests (default is True).

        Raises
        ------
        ValueError
            If the API key is not provided to the client, is not set as an environment variable or is not a string.
        """
        # Manage the API key
        if api_key is None or not isinstance(api_key, str):
            try:
                api_key = os.environ["WECO_API_KEY"]
            except KeyError:
                raise ValueError("WECO_API_KEY must be passed to client or set as an environment variable")
        self.api_key = api_key
        self.http2 = http2
        self.timeout = timeout
        self.base_url = "https://function.api.weco.ai"
        # Setup clients
        self.client = httpx.Client(http2=http2, timeout=timeout)
        self.async_client = httpx.AsyncClient(http2=http2, timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        """Constructs the headers for the API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _make_request(self, endpoint: str, data: Dict[str, Any], is_async: bool = False) -> Callable:
        """Creates a callable for making either synchronous or asynchronous requests.

        Parameters
        ----------
        endpoint : str
            The API endpoint to which the request will be made.
        data : dict
            The data to be sent in the request body.
        is_async : bool, optional
            Whether to create an asynchronous request (default is False).

        Returns
        -------
        Callable
            A callable that performs the HTTP request.
        """
        url = f"{self.base_url}/{endpoint}"
        headers = self._headers()

        if is_async:

            async def _request():
                try:
                    response = await self.async_client.post(url, json=data, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except HTTPStatusError as e:
                    # Handle HTTP errors (4xx and 5xx status codes)
                    error_message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
                    raise ValueError(error_message) from e
                except Exception as e:
                    # Handle other exceptions
                    raise ValueError(f"An error occurred: {str(e)}") from e

            return _request()
        else:

            def _request():
                try:
                    response = self.client.post(url, json=data, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except HTTPStatusError as e:
                    # Handle HTTP errors (4xx and 5xx status codes)
                    error_message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
                    raise ValueError(error_message) from e
                except Exception as e:
                    # Handle other exceptions
                    raise ValueError(f"An error occurred: {str(e)}") from e

            return _request()

    def _process_query_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the query response and handles warnings.

        Parameters
        ----------
        response : dict
            The raw API response.

        Returns
        -------
        dict
            A processed dictionary containing the output, token counts, and latency.

        Raises
        ------
        UserWarning
            If there are any warnings in the API response.
        """
        for _warning in response.get("warnings", []):
            warnings.warn(_warning)

        returned_response = {
            "output": response["response"],
            "in_tokens": response["num_input_tokens"],
            "out_tokens": response["num_output_tokens"],
            "latency_ms": response["latency_ms"],
        }
        if "reasoning_steps" in response:
            returned_response["reasoning_steps"] = response["reasoning_steps"]
        return returned_response

    def _build(
        self, task_description: str, multimodal: bool, is_async: bool
    ) -> Union[Tuple[str, int, str], Coroutine[Any, Any, Tuple[str, int, str]]]:
        """Internal method to handle both synchronous and asynchronous build requests.

        Parameters
        ----------
        task_description : str
            A description of the task for which the function is being built.

        multimodal : bool
            Whether the function is multimodal or not.

        is_async : bool
            Whether to perform an asynchronous request.

        Returns
        -------
        Union[tuple[str, int, str], Coroutine[Any, Any, tuple[str, int, str]]]
            A tuple containing the name, version number and description of the function, or a coroutine that returns such a tuple.

        Raises
        ------
        ValueError
            If the task description is empty or exceeds the maximum length.
        """
        # Validate the input
        if len(task_description) == 0:
            raise ValueError("Task description must be provided.")
        if len(task_description) > MAX_TEXT_LENGTH:
            raise ValueError(f"Task description must be less than {MAX_TEXT_LENGTH} characters.")

        endpoint = "build"
        data = {"request": task_description, "multimodal": multimodal}
        request = self._make_request(endpoint=endpoint, data=data, is_async=is_async)

        if is_async:
            # return 0 for the version number
            async def _async_build():
                response = await request
                return response["function_name"], 0, response["description"]

            return _async_build()
        else:
            response = request  # the request has already been made and the response is available
            return response["function_name"], 0, response["description"]

    async def abuild(self, task_description: str, multimodal: bool = False) -> Tuple[str, int, str]:
        """Asynchronously builds a specialized function given a task description.

        Parameters
        ----------
        task_description : str
            A description of the task for which the function is being built.

        multimodal : bool, optional
            Whether the function is multimodal or not (default is False).

        Returns
        -------
        tuple[str, str]
            A tuple containing the name, version number and description of the function.
        """
        return await self._build(task_description=task_description, multimodal=multimodal, is_async=True)

    def build(self, task_description: str, multimodal: bool = False) -> Tuple[str, int, str]:
        """Synchronously builds a specialized function given a task description.

        Parameters
        ----------
        task_description : str
            A description of the task for which the function is being built.

        multimodal : bool, optional
            Whether the function is multimodal or not (default is False).

        Returns
        -------
        tuple[str, str]
            A tuple containing the name, version number and description of the function.
        """
        return self._build(task_description=task_description, multimodal=multimodal, is_async=False)

    def _upload_image(self, fn_name: str, version_number: int, upload_id: str, image_info: Dict[str, Any]) -> str:
        """
        Uploads an image to an S3 bucket and returns the URL of the uploaded image.

        Parameters
        ----------
        fn_name : str
            The name of the function for which the image is being uploaded.
        version_number : int
            The version number of the function for which the image is being uploaded.
        upload_id: str
            A unique identifier for the image upload.
        image_info : Dict[str, Any]
            A dictionary containing the image metadata.

        Returns
        -------
        str
            The URL of the uploaded image.
        """

        if image_info["source"] == "base64":
            _, base64_info = is_base64_image(maybe_base64=image_info["image"])
            img_data = base64.b64decode(base64_info["encoding"])
        elif image_info["source"] == "url":
            response = requests.get(image_info["image"])
            response.raise_for_status()
            img_data = response.content
        elif image_info["source"] == "local":
            with open(image_info["image"], "rb") as f:
                img_data = f.read()
        else:
            raise ValueError("Invalid image input")

        # Preprocess the image
        img = Image.open(BytesIO(img_data))
        file_type = image_info["file_type"]
        processed_img, file_type = preprocess_image(image=img, file_type=file_type)
        upload_data = BytesIO()
        processed_img.save(upload_data, format=file_type)
        upload_data = upload_data.getvalue()

        # Request a presigned URL from the server
        endpoint = "upload_link"
        request_data = {"fn_name": fn_name, "version_number": version_number, "upload_id": upload_id, "file_type": file_type}
        # This needs to be a synchronous request since we need the presigned URL to upload the image
        response = self._make_request(endpoint=endpoint, data=request_data, is_async=False)

        # Upload the image to the S3 bucket
        files = {"file": upload_data}
        http_response = requests.post(response["url"], data=response["fields"], files=files)
        if http_response.status_code == 204:
            pass
        else:
            raise ValueError("Image upload failed")

        # Return the URL of the uploaded image
        upload_link = f"{response['url']}{response['fields']['key']}"
        return upload_link

    def _validate_query(self, text_input: str, images_input: List[str]) -> List[Dict[str, Any]]:
        """
        Validate the input for the query method.

        Parameters
        ----------
        text_input : str
            The text input to the function.
        images_input : List[str]
            A list of image URLs or images encoded in base64 with their metadata to be sent as input to the function.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries containing the image metadata.

        Raises
        ------
        ValueError
            If the input is invalid.
        """
        if not isinstance(text_input, str) or not isinstance(images_input, list):
            raise ValueError("Text input must be a string and images input must be a list of strings.")
        for image in images_input:
            if not isinstance(image, str):
                raise ValueError("Images input must be a list of strings.")

        # Assert that either text or images or both must be provded
        if len(text_input) == 0 and len(images_input) == 0:
            raise ValueError("Either text or images or both must be provided as input.")

        # Check if the text input is within the limit
        if len(text_input) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text input must be less than {MAX_TEXT_LENGTH} characters.")

        # Check if the images input is within the limit
        if len(images_input) > MAX_IMAGE_UPLOADS:
            raise ValueError(f"Number of images must be less than {MAX_IMAGE_UPLOADS}.")

        # Check if input is an valid image
        image_info = []
        for image in images_input:
            is_base64, base64_info = is_base64_image(maybe_base64=image)
            if is_base64:
                file_type = base64_info["media_type"].split("/")[1]

            is_public_url = is_public_url_image(maybe_url_image=image)
            if is_public_url:
                response = requests.get(image)
                response.raise_for_status()
                file_type = response.headers["content-type"].split("/")[1]

            is_local = is_local_image(maybe_local_image=image)
            if is_local:
                file_type = os.path.splitext(image)[1][1:]

            if not (is_base64 or is_public_url or is_local):
                raise ValueError("Images must be local paths, public URLs or base64 encoded strings.")

            # Determine the source of image
            if is_base64:
                source = "base64"
            elif is_public_url:
                source = "url"
            elif is_local:
                source = "local"

            # Check if the image type is supported
            file_type = file_type.lower()
            if file_type not in SUPPORTED_IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Image file type {file_type} is not supported. Supported types are {SUPPORTED_IMAGE_EXTENSIONS}."
                )

            # Check if the image size is within the limit
            size = get_image_size(image=image, source=source)
            if size > MAX_IMAGE_SIZE_MB:
                raise ValueError(f"Individual image sizes must be less than {MAX_IMAGE_SIZE_MB} MB each.")

            image_info.append({"image": image, "file_type": file_type, "size": size, "source": source})

        return image_info

    def _query(
        self,
        is_async: bool,
        fn_name: str,
        version_number: Optional[int],
        text_input: Optional[str],
        images_input: Optional[List[str]],
        return_reasoning: Optional[bool]
    ) -> Union[Dict[str, Any], Coroutine[Any, Any, Dict[str, Any]]]:
        """Internal method to handle both synchronous and asynchronous query requests.

        Parameters
        ----------
        is_async : bool
            Whether to perform an asynchronous request.
        fn_name : str
            The name of the function to query.
        version_number : int, optional
            The version number of the function to query.
        text_input : str, optional
            The text input to the function.
        images_input : List[str], optional
            A list of image URLs or images encoded in base64 with their metadata to be sent as input to the function.
        return_reasoning : bool, optional
            Whether to return reasoning for the output.

        Returns
        -------
        Union[Dict[str, Any], Coroutine[Any, Any, dict]]
            A dictionary containing the query results, or a coroutine that returns such a dictionary.

        Raises
        ------
        ValueError
            If the input is invalid.
        """
        # Validate the input
        image_info = self._validate_query(text_input=text_input, images_input=images_input)

        # Create links for all images that are not public URLs and upload images
        image_urls = []
        upload_id = generate_random_base16_code()
        for i, info in enumerate(image_info):
            if info["source"] == "url" or info["source"] == "base64" or info["source"] == "local":
                url = self._upload_image(fn_name=fn_name, version_number=version_number, upload_id=upload_id, image_info=info)
            else:
                raise ValueError(f"Image at index {i} must be a public URL or a path to a local image file.")
            image_urls.append(url)

        # Make the request
        endpoint = "query"
        data = {"name": fn_name, "text": text_input, "images": image_urls, "version_number": version_number, "return_reasoning": return_reasoning}
        request = self._make_request(endpoint=endpoint, data=data, is_async=is_async)

        if is_async:

            async def _async_query():
                response = await request
                return self._process_query_response(response=response)

            return _async_query()
        else:
            response = request  # the request has already been made and the response is available
            return self._process_query_response(response=response)

    async def aquery(
        self,
        fn_name: str,
        version_number: Optional[int] = -1,
        text_input: Optional[str] = "",
        images_input: Optional[List[str]] = [],
        return_reasoning: Optional[bool] = False
    ) -> Dict[str, Any]:
        """Asynchronously queries a function with the given function ID and input.

        Parameters
        ----------
        fn_name : str
            The name of the function to query.
        version_number : int, optional
            The version number of the function to query. If not provided, the latest version will be used. Pass -1 to use the latest version.
        text_input : str, optional
            The text input to the function.
        images_input : List[str], optional
            A list of image URLs or images encoded in base64 with their metadata to be sent as input to the function.
        return_reasoning : bool, optional
            Whether to return reasoning for the output. Default is False.

        Returns
        -------
        dict
            A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
            and the latency in milliseconds.
        """
        return await self._query(
            fn_name=fn_name, version_number=version_number, text_input=text_input, images_input=images_input, return_reasoning=return_reasoning, is_async=True
        )

    def query(
        self,
        fn_name: str,
        version_number: Optional[int] = -1,
        text_input: Optional[str] = "",
        images_input: Optional[List[str]] = [],
        return_reasoning: Optional[bool] = False
    ) -> Dict[str, Any]:
        """Synchronously queries a function with the given function ID and input.

        Parameters
        ----------
        fn_name : str
            The name of the function to query.
        version_number : int, optional
            The version number of the function to query. If not provided, the latest version will be used. Pass -1 to use the latest version.
        text_input : str, optional
            The text input to the function.
        images_input : List[str], optional
            A list of image URLs or images encoded in base64 with their metadata to be sent as input to the function.
        return_reasoning : bool, optional
            Whether to return reasoning for the output. Default is False.

        Returns
        -------
        dict
            A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
            and the latency in milliseconds.
        """
        return self._query(
            fn_name=fn_name, version_number=version_number, text_input=text_input, images_input=images_input, return_reasoning=return_reasoning, is_async=False
        )

    def batch_query(
        self, fn_name: str, batch_inputs: List[Dict[str, Any]], version_number: Optional[int] = -1, return_reasoning: Optional[bool] = False
    ) -> List[Dict[str, Any]]:
        """Batch queries a function version with a list of inputs.

        Parameters
        ----------
        fn_name : str
            The name of the function or a list of function names to query.
        batch_inputs : List[Dict[str, Any]]
            A list of inputs for the functions to query. The input must be a dictionary containing the data to be processed. e.g.,
            when providing for a text input, the dictionary should be {"text_input": "input text"}, for an image input, the dictionary should be {"images_input": ["url1", "url2", ...]}
            and for a combination of text and image inputs, the dictionary should be {"text_input": "input text", "images_input": ["url1", "url2", ...]}.
        version_number : int, optional
            The version number of the function to query. If not provided, the latest version will be used. Pass -1 to use the latest version.
        return_reasoning : bool, optional
            Whether to return reasoning for the output. Default is False.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each containing the output of a function query,
            in the same order as the input queries.
        """

        async def run_queries():
            tasks = list(
                map(lambda fn_input: self.aquery(fn_name=fn_name, version_number=version_number, return_reasoning=return_reasoning, **fn_input), batch_inputs)
            )
            return await asyncio.gather(*tasks)

        return asyncio.run(run_queries())
