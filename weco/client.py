import asyncio
import requests
import os
import base64
from io import BytesIO
from PIL import Image
import warnings
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import httpx

from .constants import MAX_TEXT_LENGTH, MAX_IMAGE_UPLOADS, MAX_IMAGE_SIZE_MB, ALLOWED_IMAGE_EXTENSIONS
from .utils import is_public_url_image, is_base64_image, is_local_image, get_image_size, preprocess_image


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

    def upload_image(self, image: str) -> str:
        """
        Uploads an image to an S3 bucket and returns the URL of the uploaded image.

        Parameters
        ----------
        image : str
            The path, base64 encoding, or URL of the image file to upload.

        Returns
        -------
        str
            The URL of the uploaded image.
        """
        is_base64, image_info = is_base64_image(maybe_base64=image)

        if is_base64:
            img_data = base64.b64decode(image_info['encoding'])
            file_type = image_info['media_type'].split('/')[1]
        elif is_public_url_image(maybe_url_image=image):
            response = requests.get(image)
            response.raise_for_status()
            img_data = response.content
            file_type = response.headers['content-type'].split('/')[1]
        elif is_local_image(maybe_local_image=image):
            with open(image, 'rb') as f:
                img_data = f.read()
            file_type = os.path.splitext(image)[1][1:]
        else:
            raise ValueError("Invalid image input")
        
        # Preprocess the image
        img = Image.open(BytesIO(img_data))
        file_type = file_type.lower()
        processed_img = preprocess_image(image=img)

        # Request a presigned URL from the server
        endpoint = "upload_link"
        data = {"file_type": file_type}
        # This needs to be a synchronous request since we need the presigned URL to upload the image
        response = self._make_request(endpoint=endpoint, data=data, is_async=False)
        presigned_url = response["url"]

        # Upload the image to the S3 bucket
        data = BytesIO()
        processed_img.save(data, format=file_type)
        data = data.getvalue()
        headers = {
            "Content-Type": f"image/{file_type}"
        }
        response = requests.put(presigned_url, data=data, headers=headers)
        response.raise_for_status()

        # Return the URL of the uploaded image
        return presigned_url


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
        for image in images_input:
            is_base64, image_info = is_base64_image(maybe_base64=image)
            if is_base64:
                file_type = image_info['media_type'].split('/')[1]
            elif is_public_url_image(maybe_url_image=image):
                response = requests.get(image)
                response.raise_for_status()
                file_type = response.headers['content-type'].split('/')[1]
            elif is_local_image(maybe_local_image=image):
                file_type = os.path.splitext(image)[1][1:]
            else:
                raise ValueError("Images must be local paths, public URLs or base64 encoded strings.")
            
            # Check if the image type is supported
            file_type = file_type.lower()
            if file_type not in ALLOWED_IMAGE_EXTENSIONS:
                raise ValueError(f"Image file type {file_type} is not supported. Supported types are {ALLOWED_IMAGE_EXTENSIONS}.")
            
            # Check if the image size is within the limit
            if get_image_size(image) > MAX_IMAGE_SIZE_MB:
                raise ValueError(f"Individual image sizes must be less than {MAX_IMAGE_SIZE_MB} MB each.")
        
        # Create links for all images that are not public URLs
        # and upload images
        for i, image in enumerate(images_input):
            if is_public_url_image(image):
                continue
            elif is_base64_image(image)[0]:
                image_url = self.upload_image(image)
                images_input[i] = image_url
            elif is_local_image(image):
                image_url = self.upload_image(image)
                images_input[i] = image_url
            else:
                raise ValueError(f"Image at index {i} must be a public URL or a path to a local image file.")
        
        # Make the request
        endpoint = "query"
        data = {"name": fn_name, "text": text_input, "images": images_input}
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
