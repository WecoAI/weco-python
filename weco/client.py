import asyncio
import os
import warnings
from typing import Any, Callable, Coroutine, Dict, List, Tuple

import httpx


class WecoAI:
    """A client for the WecoAI function builder API that allows users to build and query specialized functions built by LLMs.
    The user must simply provide a task description to build a function, and then query the function with an input to get the result they need.
    Our client supports both synchronous and asynchronous request paradigms and uses HTTP/2 for faster communication with the API.

    Attributes
    ----------
    api_key : str
        The API key used for authentication.
    """

    def __init__(self, api_key: str = None, timeout: float = 30.0) -> None:
        """Initializes the WecoAI client with the provided API key and base URL.

        Parameters
        ----------
        api_key : str, optional
            The API key used for authentication. If not provided, the client will attempt to read it from the environment variable - WECO_API_KEY.

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

        self.base_url = "https://function-builder.vercel.app"

        # Setup clients
        self.client = httpx.Client(http2=False, timeout=timeout)
        self.async_client = httpx.AsyncClient(http2=False, timeout=timeout)

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
        """
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

    def _query(self, fn_name: str, fn_input: str, is_async: bool) -> Dict[str, Any] | Coroutine[Any, Any, Dict[str, Any]]:
        """Internal method to handle both synchronous and asynchronous query requests.

        Parameters
        ----------
        fn_name : str
            The name of the function to query.
        fn_input : str
            The input to the function.
        is_async : bool
            Whether to perform an asynchronous request.

        Returns
        -------
        dict | Coroutine[Any, Any, dict]
            A dictionary containing the query results, or a coroutine that returns such a dictionary.
        """
        endpoint = "query"
        data = {"name": fn_name, "user_message": fn_input}
        request = self._make_request(endpoint=endpoint, data=data, is_async=is_async)

        if is_async:

            async def _async_query():
                response = await request
                return self._process_response(response=response)

            return _async_query()
        else:
            response = request  # the request has already been made and the response is available
            return self._process_response(response=response)

    async def aquery(self, fn_name: str, fn_input: str) -> Dict[str, Any]:
        """Asynchronously queries a function with the given function ID and input.

        Parameters
        ----------
        fn_name : str
            The name of the function to query.
        fn_input : str
            The input to the function.

        Returns
        -------
        dict
            A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
            and the latency in milliseconds.
        """
        return await self._query(fn_name=fn_name, fn_input=fn_input, is_async=True)

    def query(self, fn_name: str, fn_input: str) -> Dict[str, Any]:
        """Synchronously queries a function with the given function ID and input.

        Parameters
        ----------
        fn_name : str
            The name of the function to query.
        fn_input : str
            The input to the function.

        Returns
        -------
        dict
               A dictionary containing the output of the function, the number of input tokens, the number of output tokens,
            and the latency in milliseconds.
        """
        return self._query(fn_name=fn_name, fn_input=fn_input, is_async=False)

    def batch_query(self, fn_names: str | List[str], batch_inputs: List[str]) -> List[Dict[str, Any]]:
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
            A list of inputs for the functions to query.
            Note that the index of each input must correspond to the index of the function name.

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
            tasks = [self.aquery(fn_name=fn_name, fn_input=fn_input) for fn_name, fn_input in zip(fn_names, batch_inputs)]
            return await asyncio.gather(*tasks)

        return asyncio.run(run_queries())
