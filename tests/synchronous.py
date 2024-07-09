import pytest
from weco import build, query


# Internally, these functions use the WecoAI client
# therefore, we can test both the client and functional forms here
@pytest.fixture
def ml_task_evaluator():
    fn_name, fn_desc = build(
        task_description="I want to evaluate the feasibility of a machine learning task. Give me a json object with three keys - 'feasibility', 'justification', and 'suggestions'."
    )
    return fn_name, fn_desc

def test_build(ml_task_evaluator):
    fn_name, fn_desc = ml_task_evaluator
    assert isinstance(fn_name, str)
    assert isinstance(fn_desc, str)

def test_query(ml_task_evaluator):
    fn_name, _ = ml_task_evaluator
    query_response = query(
        fn_name=fn_name,
        fn_input="I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle.",
    )
    
    assert isinstance(query_response, dict)
    assert isinstance(query_response["output"], dict)
    assert isinstance(query_response["in_tokens"], int)
    assert isinstance(query_response["out_tokens"], int)
    assert isinstance(query_response["latency_ms"], float)

    assert set(query_response["output"].keys()) == {"feasibility", "justification", "suggestions"}
 