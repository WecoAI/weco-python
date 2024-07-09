import pytest
from weco import abuild, aquery

# Internally, these functions use the WecoAI client
# therefore, we can test both the client and functional forms here
@pytest.fixture
async def ml_task_evaluator():
    fn_name, fn_desc = await abuild(
        task_description="I want to evaluate the feasibility of a machine learning task. Give me a json object with three keys - 'feasibility', 'justification', and 'suggestions'."
    )
    return fn_name, fn_desc

@pytest.mark.asyncio
async def test_abuild(ml_task_evaluator):
    fn_name, fn_desc = ml_task_evaluator
    assert isinstance(fn_name, str)
    assert isinstance(fn_desc, str)

@pytest.mark.asyncio
async def test_aquery(ml_task_evaluator):
    fn_name, _ = ml_task_evaluator
    query_response = await aquery(
        fn_name=fn_name,
        fn_input="I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle.",
    )
    
    assert isinstance(query_response, dict)
    assert isinstance(query_response["output"], dict)
    assert isinstance(query_response["in_tokens"], int)
    assert isinstance(query_response["out_tokens"], int)
    assert isinstance(query_response["latency_ms"], float)

    assert set(query_response["output"].keys()) == {"feasibility", "justification", "suggestions"} 
    