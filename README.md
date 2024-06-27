<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <img src="assets/weco.svg" alt="WeCo AI" style="height: 50px; margin-right: 10px;">
  <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Georgia&size=32&duration=4000&pause=400&color=FD4578&vCenter=true&multiline=false&width=200&height=50&lines=WeCo+Client" alt="Typing SVG" /></a>
</div>

# $f$(👷‍♂️)

A client facing API for interacting with the [WeCo AI](https://www.weco.ai/) function builder [service](https://weco-app.vercel.app/function)!


Use this API to build *complex* systems *fast*. We lower the barrier of entry to software engineer, data science and machine learning by providing an interface to prototype difficult solutions quickly in just a few lines of code.

## Installation

Install the `weco` package simply by calling this in your terminal of choice:
```bash
pip install weco
```

## Features

- The **build** function enables quick and easy prototyping of new functions via LLMs through just natural language. We encourage users to do this through our [web console](https://weco-app.vercel.app/function) for maximum control and ease of use, however, you can also do this through our API as shown in [here](examples/cookbook.ipynb).
- The **query** function allows you to test and use the newly created function in your own code.
- We offer asynchronous versions of the above clients.
- We provide a **batch_query** functions that allows users to batch functions for various inputs as well as multiple inputs for the same function in a query. This is helpful to make a large number of queries more efficiently.

We provide both services in two ways:
- `weco.WecoAI` client to be used when you want to maintain the same client service across a portion of code. This is better for dense service usage.
- `weco.query` and `weco.build` to be used when you only require sparse usage.

## Usage

When using the WeCo API, you will need to set the API key:
You can find/setup your API key [here](https://weco-app.vercel.app/account) by navigating to the API key tab. Once you have your API key, you may pass it to the `weco` client using the `api_key` argument input or set it as an environment variable such as:
```bash
export WECO_API_KEY=<YOUR_WECO_API_KEY>
```

## Example

We create a function on the [web console](https://weco-app.vercel.app/function) for the following task:
> "I want to evaluate the feasibility of a machine learning task. Give me a json object with three keys - 'feasibility', 'justification', and 'suggestions'."

Now, you're ready to query this function anywhere in your code!

```python
from weco import query
response = query(
    fn_name=fn_name,
    fn_input="I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle.",
)
```

For more examples and an advanced user guide, check out our function builder [cookbook](examples/cookbook.ipynb).

## Happy building $f$(👷‍♂️)!