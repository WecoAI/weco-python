import asyncio
import base64
import os
import warnings
from io import BytesIO
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import httpx
import requests
from PIL import Image

from .constants import MAX_IMAGE_SIZE_MB, MAX_IMAGE_UPLOADS, MAX_TEXT_LENGTH, SUPPORTED_IMAGE_EXTENSIONS
from .utils import get_image_size, is_base64_image, is_local_image, is_public_url_image, preprocess_image


class WecoAI:
    """A client for the WecoAI function builder API that allows users to build and query specialized functions built by LLMs.
    The user must simply provide a task description to build a function, and then query the function with an input to get the result they need.
    Our client supports both synchronous and asynchronous request paradigms and uses HTTP/2 for faster communication with the API.

    Attributes
    ----------
    api_key : str
        The API key used for authentication.
    """

    def __init__(self, api_key: str = None, timeout: float = 30.0, http2: bool = True) -> None:
        """Initializes the WecoAI client with the provided API key and base URL.

        Parameters
        ----------
        api_key : str, optional
            The API key used for authentication. If not provided, the client will attempt to read it from the environment variable - WECO_API_KEY.

        timeout : float, optional
            The timeout for the HTTP requests in seconds (default is 30.0).

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

        self.base_url = "https://function.api.weco.ai"
        # Setup clients
        self.client = httpx.Client(http2=http2, timeout=timeout)
        self.async_client = httpx.AsyncClient(http2=http2, timeout=timeout)

    def __del__(self):
        """Closes the HTTP clients when the WecoAI instance is deleted."""
        try:
            self.client.close()
            if not self.async_client.is_closed:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.async_client.aclose())
                    else:
                        loop.run_until_complete(self.async_client.aclose())
                except RuntimeError:
                    # If the event loop is closed, we can't do anything about it
                    pass
        except AttributeError:
            # If the client is not initialized, we can't do anything about it
            pass

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
                response = await self.async_client.post(url, json=data, headers=headers)
                response.raise_for_status()
                return response.json()

            return _request()
        else:

            def _request():
                response = self.client.post(url, json=data, headers=headers)
                response.raise_for_status()
                return response.json()

            return _request()

    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the API response and handles warnings.

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

        return {
            "output": response["response"],
            "in_tokens": response["num_input_tokens"],
            "out_tokens": response["num_output_tokens"],
            "latency_ms": response["latency_ms"],
        }

    def _build(self, task_description: str, is_async: bool) -> Tuple[str, str] | Coroutine[Any, Any, Tuple[str, str]]:
        """Internal method to handle both synchronous and asynchronous build requests.

        Parameters
        ----------
        task_description : str
            A description of the task for which the function is being built.
        is_async : bool
            Whether to perform an asynchronous request.

        Returns
        -------
        tuple[str, str] | Coroutine[Any, Any, tuple[str, str]]
            A tuple containing the name and description of the function, or a coroutine that returns such a tuple.

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
        data = {"request": task_description}
        request = self._make_request(endpoint=endpoint, data=data, is_async=is_async)

        if is_async:

            async def _async_build():
                response = await request
                return response["name"], response["description"]

            return _async_build()
        else:
            response = request  # the request has already been made and the response is available
            return response["name"], response["description"]

    async def abuild(self, task_description: str) -> Tuple[str, str]:
        """Asynchronously builds a specialized function given a task description.

        Parameters
        ----------
        task_description : str
            A description of the task for which the function is being built.

        Returns
        -------
        tuple[str, str]
            A tuple containing the name and description of the function.
        """
        return await self._build(task_description=task_description, is_async=True)

    def build(self, task_description: str) -> Tuple[str, str]:
        """Synchronously builds a specialized function given a task description.

        Parameters
        ----------
        task_description : str
            A description of the task for which the function is being built.

        Returns
        -------
        tuple[str, str]
            A tuple containing the name and description of the function.
        """
        return self._build(task_description=task_description, is_async=False)

    def _upload_image(self, image_info: Dict[str, Any]) -> str:
        """
        Uploads an image to an S3 bucket and returns the URL of the uploaded image.

        Parameters
        ----------
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

        # TODO: Test the next lines till the end of the function
        # Request a presigned URL from the server
        endpoint = "upload_link"
        request_data = {"file_type": file_type}
        # This needs to be a synchronous request since we need the presigned URL to upload the image
        response = self._make_request(endpoint=endpoint, data=request_data, is_async=False)
        upload_link = response["url"]

        # Upload the image to the S3 bucket
        headers = {"Content-Type": f"image/{file_type}"}
        response = requests.put(upload_link, data=upload_data, headers=headers)
        response.raise_for_status()

        # Return the URL of the uploaded image
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
                try:
                    file_type = base64_info["media_type"].split("/")[1]
                except Exception as _:
                    raise ValueError(
                        "Invalid image base64 encoding. Try providing a valid image URL, local path or base64 encoded string."
                    )

            is_public_url = is_public_url_image(maybe_url_image=image)
            if is_public_url:
                response = requests.get(image)
                response.raise_for_status()
                try:
                    file_type = response.headers["content-type"].split("/")[1]
                except Exception as _:
                    raise ValueError(
                        "Invalid image URL. Try providing a valid image URL, local path or base64 encoded string."
                    )
            
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
        self, is_async: bool, fn_name: str, text_input: Optional[str], images_input: Optional[List[str]]
    ) -> Dict[str, Any] | Coroutine[Any, Any, Dict[str, Any]]:
        """Internal method to handle both synchronous and asynchronous query requests.

        Parameters
        ----------
        is_async : bool
            Whether to perform an asynchronous request.
        fn_name : str
            The name of the function to query.
        text_input : str, optional
            The text input to the function.
        images_input : List[str], optional
            A list of image URLs or images encoded in base64 with their metadata to be sent as input to the function.

        Returns
        -------
        dict | Coroutine[Any, Any, dict]
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
        for i, info in enumerate(image_info):
            if info["source"] == "url":
                url = info["image"]
            elif info["source"] == "base64" or info["source"] == "local":
                url = self._upload_image(info)
            else:
                raise ValueError(f"Image at index {i} must be a public URL or a path to a local image file.")
            image_urls.append(url)

        # Make the request
        endpoint = "query"
        data = {"name": fn_name, "text": text_input, "images": image_urls}
        request = self._make_request(endpoint=endpoint, data=data, is_async=is_async)

        if is_async:

            async def _async_query():
                response = await request
                return self._process_response(response=response)

            return _async_query()
        else:
            response = request  # the request has already been made and the response is available
            return self._process_response(response=response)

    async def aquery(
        self, fn_name: str, text_input: Optional[str] = "", images_input: Optional[List[str]] = []
    ) -> Dict[str, Any]:
        """Asynchronously queries a function with the given function ID and input.

        Parameters
        ----------
        fn_name : str
            The name of the function to query.
        text_input : str, optional
            The text input to the function.
        images_input : List[str], optional
            A list of image URLs or images encoded in base64 with their metadata to be sent as input to the function.

        Returns
        -------
        dict
            A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
            and the latency in milliseconds.
        """
        return await self._query(fn_name=fn_name, text_input=text_input, images_input=images_input, is_async=True)

    def query(self, fn_name: str, text_input: Optional[str] = "", images_input: Optional[List[str]] = []) -> Dict[str, Any]:
        """Synchronously queries a function with the given function ID and input.

        Parameters
        ----------
        fn_name : str
            The name of the function to query.
        text_input : str, optional
            The text input to the function.
        images_input : List[str], optional
            A list of image URLs or images encoded in base64 with their metadata to be sent as input to the function.

        Returns
        -------
        dict
               A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
            and the latency in milliseconds.
        """
        return self._query(fn_name=fn_name, text_input=text_input, images_input=images_input, is_async=False)

    def batch_query(self, fn_names: str | List[str], batch_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronously queries multiple functions using asynchronous calls internally.

        This method uses the asynchronous queries to submit all queries concurrently
        and waits for all responses to be received before returning the results.

        Parameters
        ----------
        fn_name : str | List[str]
            The name of the function or a list of function names to query.
            Note that if a single function name is provided, it will be used for all queries.
            If a list of function names is provided, the length must match the number of queries.

        batch_inputs : List[Dict[str, Any]]
            A list of inputs for the functions to query. The input must be a dictionary containing the data to be processed. e.g.,
            when providing for a text input, the dictionary should be {"text_input": "input text"}, for an image input, the dictionary should be {"images_input": ["url1", "url2", ...]}
            and for a combination of text and image inputs, the dictionary should be {"text_input": "input text", "images_input": ["url1", "url2", ...]}.
            Note that the index of each input must correspond to the index of the function name when both inputs are lists.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each containing the output of a function query,
            in the same order as the input queries.


        Raises
        ------
        ValueError
            If the number of function names (when provided as a list) does not match the number of inputs.
        """
        if isinstance(fn_names, str):
            fn_names = [fn_names] * len(batch_inputs)
        elif len(fn_names) != len(batch_inputs):
            raise ValueError("The number of function names must match the number of inputs.")

        async def run_queries():
            tasks = [
                self.aquery(fn_name=fn_name, **fn_input)  # unpack the input kwargs
                for fn_name, fn_input in zip(fn_names, batch_inputs)
            ]
            return await asyncio.gather(*tasks)

        return asyncio.run(run_queries())
