<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <img src="assets/weco.svg" alt="WeCo AI" style="height: 50px; margin-right: 10px;">
  <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Georgia&size=32&duration=4000&pause=400&color=FD4578&vCenter=true&multiline=false&width=200&height=50&lines=WeCo+Client" alt="Typing SVG" /></a>
</div>

![Python](https://img.shields.io/badge/Python-3.10.14-blue)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<!-- TODO: Update examples -->
# $f$(👷‍♂️)

<a href="https://colab.research.google.com/github/WecoAI/weco-python/blob/main/examples/cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width=110 height=20/></a>
<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FWecoAI%2Fweco-python%2Fblob%2Fmain%2Fexamples%2Fcookbook.ipynb"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" width=100 height=20/></a>

A client facing API for interacting with the [WeCo AI](https://www.weco.ai/) function builder [service](https://weco-app.vercel.app/function)!


Use this API to build *complex* systems *fast*. We lower the barrier of entry to software engineer, data science and machine learning by providing an interface to prototype difficult solutions quickly in just a few lines of code.

## Installation

Install the `weco` package simply by calling this in your terminal of choice:
```bash
pip install weco
```

## Features
- Synchronous & Asynchronous client.
- Batch API
- Multimodality (Language & Vision)
- Interpretability (view the reasoning behind outputs)


## What We Offer

- The **build** function enables quick and easy prototyping of new functions via LLMs through just natural language. We encourage users to do this through our [web console](https://weco-app.vercel.app/function) for maximum control and ease of use, however, you can also do this through our API as shown in [here](examples/cookbook.ipynb).
- The **query** function allows you to test and use the newly created function in your own code.

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
> "Analyze a business idea and provide a structured evaluation. Output a JSON with 'viability_score' (0-100), 'strengths' (list), 'weaknesses' (list), and 'next_steps' (list)."

Now, you're ready to query this function anywhere in your code!

```python
from weco import query
response = query(
    fn_name="BusinessIdeaAnalyzer-XYZ123",  # Replace with your actual function name
    text_input="A subscription service for personalized, AI-generated bedtime stories for children."
)
```

For more examples and an advanced user guide, check out our function builder [cookbook](examples/cookbook.ipynb).

## Happy building $f$(👷‍♂️)!

## Contributing

We value your contributions! If you believe you can help to improve our package enabling people to build AI with AI, please contribute!

Use the following steps as a guideline to help you make contributions:

1. Download and install package from source:
   ```bash
   git clone https://github.com/WecoAI/weco-python.git
   cd weco-python
   pip install -e ".[dev]"
   ```

2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and run tests to ensure everything is working:
   
   > **Tests can be expensive to run as they make LLM requests with the API key being used so it is the developers best interests to write small and simple tests that adds coverage for a large portion of the package.**
   
   ```bash
   pytest -n auto tests
   ```

4. Commit and push your changes, then open a PR for us to view 😁

Please ensure your code follows our style guidelines (Numpy docstrings) and includes appropriate tests. We appreciate your contributions!
