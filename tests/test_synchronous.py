import pytest

from weco import build, query


# Internally, these functions use the WecoAI client
# therefore, we can test both the client and functional forms here
def test_build(text_evaluator, image_evaluator, text_and_image_evaluator):
    for evaluator in [text_evaluator, image_evaluator, text_and_image_evaluator]:
        fn_name, version_number, fn_desc = evaluator
        assert isinstance(fn_name, str)
        assert isinstance(version_number, int)
        assert isinstance(fn_desc, str)


def assert_query_response(query_response):
    assert isinstance(query_response, dict)
    assert isinstance(query_response["output"], dict)
    assert isinstance(query_response["in_tokens"], int)
    assert isinstance(query_response["out_tokens"], int)
    assert isinstance(query_response["latency_ms"], float)
    assert "reasoning_steps" not in query_response


@pytest.fixture
def text_evaluator():
    fn_name, version_number, fn_desc = build(
        task_description="Evaluate the sentiment of the given text. Provide a json object with 'sentiment' and 'explanation' keys.",
        multimodal=False,
    )
    return fn_name, version_number, fn_desc


def test_text_query(text_evaluator):
    fn_name, version_number, _ = text_evaluator
    query_response = query(fn_name=fn_name, version_number=version_number, text_input="I love this product!")

    assert_query_response(query_response)
    assert set(query_response["output"].keys()) == {"sentiment", "explanation"}


@pytest.fixture
def image_evaluator():
    fn_name, version_number, fn_desc = build(
        task_description="Describe the contents of the given images. Provide a json object with 'description' and 'objects' keys.",
        multimodal=True,
    )
    return fn_name, version_number, fn_desc


def test_image_query(image_evaluator):
    fn_name, version_number, _ = image_evaluator
    query_response = query(
        fn_name=fn_name,
        version_number=version_number,
        images_input=[
            "https://www.integratedtreatmentservices.co.uk/wp-content/uploads/2013/12/Objects-of-Reference.jpg",
            "https://t4.ftcdn.net/jpg/05/70/90/23/360_F_570902339_kNj1reH40GFXakTy98EmfiZHci2xvUCS.jpg",
        ],
    )

    assert_query_response(query_response)
    assert set(query_response["output"].keys()) == {"description", "objects"}


@pytest.fixture
def text_and_image_evaluator():
    fn_name, version_number, fn_desc = build(
        task_description="Evaluate, solve and arrive at a numerical answer for the image provided. Perform any additional things if instructed. Provide a json object with 'answer' and 'explanation' keys.",
        multimodal=True,
    )
    return fn_name, version_number, fn_desc


def test_text_and_image_query(text_and_image_evaluator):
    fn_name, version_number, _ = text_and_image_evaluator
    query_response = query(
        fn_name=fn_name,
        version_number=version_number,
        text_input="Find x and y.",
        images_input=[
            "https://i.ytimg.com/vi/cblHUeq3bkE/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLAKn3piY91QRCBzRgnzAPf7MPrjDQ"
        ],
    )

    assert_query_response(query_response)
    assert set(query_response["output"].keys()) == {"answer", "explanation"}
