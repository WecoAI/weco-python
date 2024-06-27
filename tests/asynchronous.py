import asyncio

from weco import abuild, aquery


# Internally, these functions use the WecoAI client
# therefore, we can test both the client and functional forms here
async def main():
    fn_name, fn_desc = await abuild(
        task_description="I want to evaluate the feasibility of a machine learning task. Give me a json object with three keys - 'feasibility', 'justification', and 'suggestions'."
    )
    print(f"Model Name: {fn_name}\nModel Description:\n{fn_desc}")

    query_response = await aquery(
        fn_name=fn_name,
        fn_input="I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle.",
    )
    for key, value in query_response.items():
        print(f"{key}: {value}")

    assert isinstance(query_response["output"], dict)
    assert isinstance(query_response["in_tokens"], int)
    assert isinstance(query_response["out_tokens"], int)
    assert isinstance(query_response["latency_ms"], float)


if __name__ == "__main__":
    asyncio.run(main())
