# API Reference Guide

The [`weco`](../index.md) package offers two modes of interacting with the AI function service. Each mode is specifically designed for different use cases.

| Submodule | Description | Use Cases |
| --- | --- | --- |
| [`weco.WecoAI`](client.md) | Weco AI client to build and query functions synchronously, asynchronously and in batches. | - Dense service usage<br>- Maintaing the same client instance over large portions of code |
| [`weco.functional`](functional.md) | Functional form of Weco AI client offering the same features as `weco.WecoAI`. | - Sparse service usage<br>- Quick prototyping |
