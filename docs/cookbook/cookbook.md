<div align="center" style="display: flex; align-items: center; justify-content: left;">
  <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Georgia&size=32&duration=4000&pause=400&color=808080&vCenter=true&multiline=false&width=750&height=100&lines=AI+Function+Cookbook;" alt="Typing SVG" /></a>
</div>

## Getting Started

`weco` is a client facing API for interacting with the [Weco AI](https://www.weco.ai/) function builder [service](https://weco-app.vercel.app/function). Use this API to build *complex* systems *fast*!

Here are a few features our users often ask about. Feel free to follow along:

<a href="https://colab.research.google.com/github/WecoAI/weco-python/blob/main/examples/cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FWecoAI%2Fweco-python%2Fblob%2Fmain%2Fexamples%2Fcookbook.ipynb"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" width=100 height=20/></a>


```python
# Install the package
%pip install weco
```

Export your API key found [here](https://www.aifunction.com/account/api-keys).


```python
%env WECO_API_KEY=<YOUR_WECO_API_KEY>
```

You can build powerful AI functions for complex tasks quickly and without friction. For example, you can create a function in the [web console](https://www.aifunction.com/function/new) with a simple description as shown below:

> "Analyze a business idea and provide a structured evaluation. Output a JSON with 'viability_score' (0-100), 'strengths' (list), 'weaknesses' (list), and 'next_steps' (list)."

Once the function has been built, you can query, test and deploy it anywhere in your code with just a few lines:


```python
from weco import query

response = query(
    fn_name="BusinessIdeaAnalyzer-XYZ123",  # Replace with your actual function name
    text_input="A subscription service for personalized, AI-generated bedtime stories for children."
)

print(response)
```

## Multimodality

Our AI functions can interpret complex visual information, follow instructions in natural language and provide practical insights. We accept a variety of different forms of image input:
1. Base64 encoding
2. Public URL
3. Local Path

Let's explore how we can have an AI function manage a part of our household. By running this once a month, I am able to find ways to cut down my energy consumption and ultimately save me money!


```python
from weco import build, query
import base64

task_description = """
You are a smart home energy analyzer that can process images of smart meters, home exteriors, 
and indoor spaces to provide energy efficiency insights. The analyzer should:
    1. Interpret smart meter readings
    2. Assess home features relevant to energy consumption
    3. Analyze thermostat settings
    4. Provide energy-saving recommendations
    5. Evaluate renewable energy potential

The output should include:
    - 'energy_consumption': current usage and comparison to average
    - 'home_analysis': visible energy features and potential issues
    - 'thermostat_settings': current settings and recommendations
    - 'energy_saving_recommendations': actionable suggestions with estimated savings
    - 'renewable_energy_potential': assessment of current and potential renewable energy use
    - 'estimated_carbon_footprint': current footprint and potential reduction
"""

fn_name, _ = build(task_description=task_description)

request = """
Analyze these images of my home and smart meter to provide energy efficiency insights 
and recommendations for reducing my electricity consumption.
"""

# Base64 encoded image
with open("/path/to/home_exterior.jpeg", "rb") as img_file:
    my_home_exterior = base64.b64encode(img_file.read()).decode('utf-8')

query_response = query(
    fn_name=fn_name,
    text_input=request,
    images_input=[
        "https://example.com/my_smart_meter_reading.png",  # Public URL
        f"data:image/jpeg;base64,{my_home_exterior}",      # Base64 encoding
        "/path/to/living_room_thermostat.jpg"              # Local image path
    ]
)

for key, value in query_response["output"].items():
    print(f"{key}: {value}")
```

## Running Example

Let's explore what an AI function can do for you and what features we offer through this running example:
> "I want to know if AI can solve a problem for me, how easy it is to arrive at a solution and whether any helpful tips for me along the way. Help me understand this through - 'feasibility', 'justification', and 'suggestions'."


```python
task_description = "I want to know if AI can solve a problem for me, how easy it is to arrive at a solution and whether any helpful tips for me along the way. Help me understand this through - 'feasibility', 'justification', and 'suggestions'."
```

## Dense vs. Sparse Usage

We recommend building functions in our [web console](https://www.aifunction.com/) for maximum control over the function with the ability to rapidly prototype, test and improve the it. However, you can also do build a function programmatically.


```python
from weco import build, query

# Describe the task you want the function to perform
fn_name, fn_desc = build(task_description=task_description)
print(f"AI Function {fn_name} built. This does the following - \n{fn_desc}.")

# Query the function with a specific input
query_response = query(
    fn_name=fn_name,
    text_input="I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle."
)
for key, value in query_response["output"].items(): print(f"{key}: {value}")
```

We recommend to use the `weco.build` and `weco.query` when you want to build or query LLM functions sparsely, i.e., you **don't** call `weco.build` or `weco.query` in many places within your code. However, for more dense usage, we've found users prefer our `weco.WecoAI` client instance. Its easy to switch between the two as shown below:


```python
from weco import WecoAI

# Connect to our service, using our client
client = WecoAI()

# Make the same query as before
query_response = client.query(
    fn_name=fn_name,
    text_input="I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle."
)
for key, value in query_response.items(): print(f"{key}: {value}")
```

## Batching

We understand that sometimes, independent of how many times in your code you call `weco` functions, you want to submit a large batch of requests for the same function you've created. This can be done in the following ways:


```python
from weco import batch_query

# Query the same function with multiple inputs by batching them for maximum efficiency
input_1 = {"text_input": "I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle."}
input_2 = {
    "text_input": "I want to train a model to classify digits using the MNIST dataset hosted on Kaggle using a Google Colab notebook. Attached is an example of what some of the digits would look like.",
    "images_input": ["https://machinelearningmastery.com/wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset-1024x768.png"]
}
query_responses = batch_query(
    fn_names=fn_name,
    batch_inputs=[input_1, input_2]
)
for i, query_response in enumerate(query_responses):
    print("-"*50)
    print(f"For input {i + 1}")
    for key, value in query_response["output"].items(): print(f"{key}: {value}")
    print("-"*50)
```

You can do the same using the `weco.WecoAI` client.

## Async Calls

Until now you've been making synchronous calls to our client by we also support asynchronous programmers. This is actually how we implement batching! You can also make asynchronous calls to our service using our `weco.WecoAI` client or as shown below for the same example as before:


```python
from weco import abuild, aquery

# Describe the task you want the function to perform
fn_name, fn_desc = await abuild(task_description=task_description)
print(f"AI Function {fn_name} built. This does the following - \n{fn_desc}.")

# Query the function with a specific input
query_response = await aquery(
    fn_name=fn_name,
    text_input="I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle."
)
for key, value in query_response["output"].items(): print(f"{key}: {value}")
```

## Interpretability

You can now understand why a model generated an output. You'll need to enable Chain of Thought (CoT) for the function version in the [web console](https://www.aifunction.com). You can find this under the **Settings** for a particular function version. Then, to view the model's reasoning behind an output, simply use `return_reasoning=True` at query time!


```python
from weco import build, query

# Describe the task you want the function to perform
fn_name, fn_desc = build(task_description=task_description)
print(f"AI Function {fn_name} built. This does the following - \n{fn_desc}.")

# Query the function with a specific input
query_response = query(
    fn_name=fn_name,
    text_input="I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle.",
    return_reasoning=True
)
for key, value in query_response["output"].items(): print(f"{key}: {value}")
for i, step in enumerate(query_response["reasoning_steps"]): print(f"Step {i+1}: {step}")
```
