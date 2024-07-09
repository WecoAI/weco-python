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
        "I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle.",
        "I want to train a model to classify digits using the MNIST dataset hosted on Kaggle using a Google Colab notebook.",
    ]

def test_batch_query(ml_task_evaluator, ml_task_inputs):
    query_responses = batch_query(fn_names=ml_task_evaluator, batch_inputs=ml_task_inputs)
    
    assert len(query_responses) == len(ml_task_inputs)
    
    for query_response in query_responses:
        assert isinstance(query_response["output"], dict)
        assert isinstance(query_response["in_tokens"], int)
        assert isinstance(query_response["out_tokens"], int)
        assert isinstance(query_response["latency_ms"], float)

        output = query_response["output"]
        assert set(output.keys()) == {"feasibility", "justification", "suggestions"}
