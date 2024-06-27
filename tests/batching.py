from weco import batch_query, build

# Internally, these functions use the WecoAI client
# therefore, we can test both the client and functional forms here
# We do NOT need to test for a list of function names here since interally
# we cast the input to a list of function names to match the fn_input size
if __name__ == "__main__":
    fn_name, fn_desc = build(
        task_description="I want to evaluate the feasibility of a machine learning task. Give me a json object with three keys - 'feasibility', 'justification', and 'suggestions'."
    )
    print(f"Model Name: {fn_name}\nModel Description:\n{fn_desc}")

    inputs = [
        "I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle.",
        "I want to train a model to classify digits using the MNIST dataset hosted on Kaggle using a Google Colab notebook.",
    ]
    query_responses = batch_query(fn_names=fn_name, batch_inputs=inputs)
    for request, response in zip(inputs, query_responses):
        print(f"Request: {request}\nResponse:")
        for key, value in response.items():
            print(f"{key}: {value}")
        print("=" * 50)

    for query_response in query_responses:
        assert isinstance(query_response["output"], dict)
        assert isinstance(query_response["in_tokens"], int)
        assert isinstance(query_response["out_tokens"], int)
        assert isinstance(query_response["latency_ms"], float)
