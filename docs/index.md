<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Georgia&size=32&duration=4000&pause=400&color=808080&vCenter=true&multiline=false&width=200&height=50&lines=Weco+Client" alt="Typing SVG" /></a>
</div>

A client facing API for interacting with the [Weco AI](https://www.weco.ai/) function builder [service](https://www.aifunction.com)!

Use this API to build *complex* systems *fast*. We lower the barrier of entry to software engineer, data science and machine learning by providing an interface to prototype difficult solutions quickly in just a few lines of code.

## What We Offer

- The **build** function enables quick and easy prototyping of new functions that use foundational models through just natural language. We encourage users to do this through our [web console](https://www.aifunction.com) for maximum control and ease of use, however, you can also do this through our API as shown [here](cookbook/cookbook.md).
- The **query** function allows you to test your newly created functions and deploy it in your code.

We provide both services in two ways:

- [`weco.WecoAI`](api/client.md) client to be used when you want to maintain the same client service across a portion of code. This is better for dense service usage or in an object oriented paradigm.
- [`weco.query`](api/functional.md) and [`weco.build`](api/functional.md) to be used when you only require sparse usage or a functional paradigm.

Some of the key features we provide are:
- Structured Output
- Grounding (Web Access)
- Multimodal (Language & Vision)
- Versatile Client (Synchronous, Asynchronous, Batch Processing)
- Interpretable (Observe Reasoning Behind Outputs)

## Getting Started

Install the [`weco`](index.md) package simply by calling this in your terminal of choice:
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
For more examples and an advanced user guide, check out our function builder [cookbook](cookbook/cookbook.md).

Happy building $f$(👷‍♂️)!
