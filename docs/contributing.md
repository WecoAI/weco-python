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
