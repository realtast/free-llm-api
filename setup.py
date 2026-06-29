#!/usr/bin/env python3
"""
Setup script for Free LLM API
"""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="free-llm-api",
    version="1.0.0",
    description="A comprehensive collection of free LLM API providers with their capabilities, limitations, and best use cases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="realtast",
    author_email="",
    url="https://github.com/realtast/free-llm-api",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "httpx>=0.25.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "anyio>=4.0.0",
        "ratelimit>=2.2.1",
        "backoff>=2.2.1",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "local-inference": [
            "torch>=2.0.0",
            "transformers>=4.37.0",
            "accelerate>=0.25.0",
            "diffusers>=0.25.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.12.0",
            "mypy>=1.5.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "llm", "ai", "machine-learning", "api", "free", "providers",
        "google", "groq", "mistral", "openrouter", "huggingface",
        "image-generation", "speech-to-text", "text-to-speech", "embeddings"
    ],
    project_urls={
        "Bug Reports": "https://github.com/realtast/free-llm-api/issues",
        "Source": "https://github.com/realtast/free-llm-api",
    },
)