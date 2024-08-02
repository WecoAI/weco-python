import pytest

from weco import batch_query, build


# Internally, these functions use the WecoAI client
# therefore, we can test both the client and functional forms here
# We do NOT need to test for a list of function names here since interally
# we cast the input to a list of function names to match the fn_input size
@pytest.fixture
def ml_task_evaluator():
    fn_name, _ = build(
        task_description="I want to evaluate the feasibility of a machine learning task. Give me a json object with three keys - 'feasibility', 'justification', and 'suggestions'."
    )
    return fn_name


@pytest.fixture
def ml_task_inputs():
    return [
        {"text_input": "I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle."},
        {
            "text_input": "I want to train a model to classify digits using the MNIST dataset hosted on Kaggle using a Google Colab notebook."
        },
    ]


@pytest.fixture
def image_evaluator():
    fn_name, _ = build(
        task_description="Describe the contents of the given images. Provide a json object with 'description' and 'objects' keys."
    )
    return fn_name


@pytest.fixture
def image_inputs():
    return [
        {
            "images_input": [
                "https://www.integratedtreatmentservices.co.uk/wp-content/uploads/2013/12/Objects-of-Reference.jpg"
            ]
        },
        {"images_input": ["https://t4.ftcdn.net/jpg/05/70/90/23/360_F_570902339_kNj1reH40GFXakTy98EmfiZHci2xvUCS.jpg"]},
    ]


def test_batch_query(ml_task_evaluator, ml_task_inputs, image_evaluator, image_inputs):
    fn_names = [ml_task_evaluator, ml_task_evaluator, image_evaluator, image_evaluator]
    batch_inputs = ml_task_inputs + image_inputs

    query_responses = batch_query(fn_names=fn_names, batch_inputs=batch_inputs)

    assert len(query_responses) == len(batch_inputs)

    for i, query_response in enumerate(query_responses):
        assert isinstance(query_response["output"], dict)
        assert isinstance(query_response["in_tokens"], int)
        assert isinstance(query_response["out_tokens"], int)
        assert isinstance(query_response["latency_ms"], float)

        output = query_response["output"]
        if i < len(ml_task_inputs):
            assert set(output.keys()) == {"feasibility", "justification", "suggestions"}
        else:
            assert set(output.keys()) == {"description", "objects"}
