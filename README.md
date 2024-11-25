<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <img src=".overrides/.icons/custom/weco.svg" alt="Weco AI" style="height: 50px; margin-right: 10px;">
  <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Georgia&size=32&duration=4000&pause=400&color=FD4578&vCenter=true&multiline=false&width=200&height=50&lines=Weco+Client" alt="Typing SVG" /></a>
</div>

![Python](https://img.shields.io/badge/Python-3.10.14-blue)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# $f$(👷‍♂️)

## IMPORTANT

This package has been deprecated. Please move to our new package - [aifn](https://github.com/WecoAI/aifn-python)!

<a href="https://colab.research.google.com/github/WecoAI/weco-python/blob/main/examples/cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width=110 height=20/></a>
<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FWecoAI%2Fweco-python%2Fblob%2Fmain%2Fexamples%2Fcookbook.ipynb"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" width=100 height=20/></a>

A client facing API for interacting with the [Weco AI](https://www.weco.ai/) function builder [service](https://www.aifunction.com)!

Use this API to build *complex* systems *fast*. We lower the barrier of entry to software engineer, data science and machine learning by providing an interface to prototype difficult solutions quickly in just a few lines of code.

## What We Offer

- The **build** function enables quick and easy prototyping of new functions that use foundational models through just natural language. We encourage users to do this through our [web console](https://www.aifunction.com) for maximum control and ease of use, however, you can also do this through our API as shown [here](examples/cookbook.ipynb).
- The **query** function allows you to test your newly created functions and deploy it in your code.

We provide both services in two ways:
- `weco.WecoAI` client to be used when you want to maintain the same client service across a portion of code. This is better for dense service usage or in an object oriented paradigm.
- `weco.query` and `weco.build` to be used when you only require sparse usage or a functional paradigm.

### Features
- Structured Output
- Grounding (Web Access)
- Multimodal (Language & Vision)
- Versatile Client (Synchronous, Asynchronous, Batch Processing)
- Interpretable (Observe Reasoning Behind Outputs)

## Getting Started

Install the `weco` package simply by calling this in your terminal of choice:
```bash
pip install weco
```

When using the Weco API, you will need to set the API key:
You can find/setup your API key [here](https://www.aifunction.com/account/api-keys). Once you have your API key, pass it directly to the client using the `api_key` argument or set it as an environment variable as shown:
```bash
export WECO_API_KEY=<YOUR_WECO_API_KEY>
```

### Example

We created a function on the [web console](https://www.aifunction.com) for the following task:
> "Analyze a business idea and provide a structured evaluation. Output a JSON with 'viability_score' (0-100), 'strengths' (list), 'weaknesses' (list), and 'next_steps' (list)."

Here's how you can query this function anywhere in your code!
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
   pip install -e ".[dev,docs]"
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
   If you're just making changes to the docs, feel free to skip this step.

4. Commit and push your changes, then open a PR for us to view 😁

Please ensure your code follows our style guidelines (Numpy docstrings) and includes appropriate tests. We appreciate your contributions!
