import pytest

from weco import build, query


def assert_query_response(query_response):
    assert isinstance(query_response, dict)
    assert isinstance(query_response["output"], dict)
    assert isinstance(query_response["reasoning_steps"], list)
    for step in query_response["reasoning_steps"]: assert isinstance(step, str)
    assert isinstance(query_response["in_tokens"], int)
    assert isinstance(query_response["out_tokens"], int)
    assert isinstance(query_response["latency_ms"], float)


@pytest.fixture
def text_reasoning_evaluator():
    fn_name, version_number, fn_desc = build(
        task_description="Evaluate the sentiment of the given text. Provide a json object with 'sentiment' and 'explanation' keys.",
        multimodal=False,
    )
    return fn_name, version_number, fn_desc


def test_text_reasoning_query(text_reasoning_evaluator):
    fn_name, version_number, _ = text_reasoning_evaluator
    query_response = query(fn_name=fn_name, version_number=version_number, text_input="I love this product!", return_reasoning=True)

    assert_query_response(query_response)
    assert set(query_response["output"].keys()) == {"sentiment", "explanation"}

@pytest.fixture
def vision_reasoning_evaluator():
    fn_name, version_number, fn_desc = build(
        task_description="Evaluate, solve and arrive at a numerical answer for the image provided. Perform any additional things if instructed. Provide a json object with 'answer' and 'explanation' keys.",
        multimodal=True,
    )
    return fn_name, version_number, fn_desc


def test_vision_reasoning_query(vision_reasoning_evaluator):
    fn_name, version_number, _ = vision_reasoning_evaluator
    query_response = query(
        fn_name=fn_name,
        version_number=version_number,
        text_input="Find x and y.",
        images_input=[
            "https://i.ytimg.com/vi/cblHUeq3bkE/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLAKn3piY91QRCBzRgnzAPf7MPrjDQ"
        ],
        return_reasoning=True,
    )

    assert_query_response(query_response)
    assert set(query_response["output"].keys()) == {"answer", "explanation"}
