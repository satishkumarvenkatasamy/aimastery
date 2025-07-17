# AI Mastery Series - Machine Learning Fundamentals

## Overview

This repository contains three comprehensive Jupyter notebooks that explore the fundamental concepts of machine learning and natural language processing. The notebooks progress from basic mathematical principles to advanced tokenization techniques used in modern AI systems.

## Notebooks

### 1. AIMastery-Session01.ipynb
**Topic**: Mathematics Behind Machine Learning with Visualizations

**Description**: This notebook provides a comprehensive introduction to the mathematical foundations of machine learning, featuring interactive visualizations and step-by-step explanations of how machines learn.

**Key Topics Covered**:
- **Linear Regression**: Understanding the foundation of ML with the equation y = mx + c
- **Cost Functions**: Sum of Squared Errors (SSE) and Mean Squared Error (MSE)
- **Gradient Descent**: The optimization algorithm that enables machine learning
- **Learning Rate**: How to control the step size in optimization
- **3D Visualizations**: Interactive plots showing error surfaces and optimization paths
- **Animated Learning**: GIF generation showing how the algorithm converges
- **Advanced Examples**: Neural networks, SVMs, and activation functions

**Highlights**:
- Complete installation guide for beginners
- Step-by-step gradient descent implementation
- Beautiful 3D visualizations of the optimization landscape
- Real-world applications including handwritten digit recognition
- Mathematical foundations using SymPy for symbolic computation

**Learning Outcomes**:
- Understand how machines learn through iterative optimization
- Visualize the relationship between parameters and error
- Grasp the mathematical principles behind modern AI systems

### 2. WordsVsTokens-Embeddings.ipynb
**Topic**: Why Tokens Are Used Over Words (Detailed Explanation)

**Description**: This notebook explains the fundamental concept of tokenization in natural language processing and why modern AI systems break words into smaller units called tokens.

**Key Topics Covered**:
- **The Million-Word Problem**: Why word-level embeddings are inefficient
- **Tokenization Benefits**: Memory efficiency, better generalization, handling unknown words
- **Subword Reusability**: How tokens like "un-", "break", and "-able" appear across multiple words
- **Semantic Clustering**: How similar tokens naturally group together
- **Vocabulary Compression**: Reducing millions of words to ~30K tokens
- **Interactive Visualizations**: 3D plots showing token relationships in embedding space

**Highlights**:
- Clear explanation of why transformers use tokenization
- Practical examples showing token reuse (e.g., "unbreakable" → ["un", "break", "able"])
- Simple tokenizer implementation without requiring heavy libraries
- Beautiful 3D visualizations of token clustering
- Real-world efficiency comparisons

**Learning Outcomes**:
- Understand why modern AI doesn't use whole words
- Learn how tokenization enables efficient language processing
- Visualize how tokens cluster in semantic space

### 3. WordsTokensUsingPytorch.ipynb
**Topic**: PyTorch Implementation of Tokenization

**Description**: This notebook demonstrates the same tokenization concepts as the previous notebook but implemented using PyTorch and the Transformers library, showing real-world tokenization in action.

**Key Topics Covered**:
- **BERT Tokenizer**: Using pre-trained tokenizers from Hugging Face
- **Real Token Embeddings**: Extracting actual embeddings from BERT
- **3D Visualization**: Interactive plots using real transformer embeddings
- **Dimensionality Reduction**: PCA and t-SNE for visualizing high-dimensional embeddings
- **Token Analysis**: Frequency analysis and categorization
- **Practical Implementation**: Working with real NLP models

**Highlights**:
- Hands-on experience with PyTorch and Transformers
- Real BERT tokenization and embeddings
- Professional-grade visualization with Plotly
- Comprehensive error handling and fallback options
- Production-ready code examples

**Learning Outcomes**:
- Practical experience with modern NLP libraries
- Understanding of real transformer tokenization
- Skills for implementing tokenization in projects

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AIMastery
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv aimastery_env
   source aimastery_env/bin/activate  # On Windows: aimastery_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # For AIMastery-Session01.ipynb
   pip install pandas numpy matplotlib seaborn sympy scikit-learn statsmodels imageio
   
   # For WordsVsTokens-Embeddings.ipynb
   pip install plotly pandas numpy scikit-learn
   
   # For WordsTokensUsingPytorch.ipynb
   pip install torch transformers plotly pandas scikit-learn matplotlib nltk
   ```

4. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

### Recommended Learning Path

1. **Start with AIMastery-Session01.ipynb** - Learn the mathematical foundations
2. **Move to WordsVsTokens-Embeddings.ipynb** - Understand tokenization concepts
3. **Complete with WordsTokensUsingPytorch.ipynb** - See real-world implementation

## Key Concepts Explained

### Why This Series Matters

Modern AI systems like GPT, BERT, and other transformers all rely on the fundamental concepts covered in these notebooks:

1. **Mathematical Optimization**: Every AI model uses gradient descent or similar optimization techniques
2. **Tokenization**: All language models break text into tokens for efficiency
3. **Embeddings**: Vector representations are the foundation of modern NLP

### From Simple to Complex

The progression shows how:
- **Simple linear regression** scales to **complex neural networks**
- **Basic tokenization** enables **powerful language models**
- **Mathematical principles** remain constant across all AI systems

### Practical Applications

These concepts are used in:
- **Language Models**: GPT, BERT, LLaMA
- **Machine Translation**: Google Translate, DeepL
- **Content Generation**: ChatGPT, Claude, Gemini
- **Search Engines**: Semantic search, question answering
- **Recommendation Systems**: Content personalization

## Visualizations

Each notebook includes rich visualizations:
- **3D optimization landscapes** showing gradient descent paths
- **Interactive token clustering** in embedding space
- **Animated learning processes** showing convergence
- **Error surface visualizations** for understanding optimization

## Technical Details

### AIMastery-Session01.ipynb
- **Language**: Python with NumPy, Pandas, Matplotlib
- **Concepts**: Linear algebra, calculus, optimization
- **Visualizations**: 3D plots, animations, contour maps
- **Difficulty**: Beginner to Intermediate

### WordsVsTokens-Embeddings.ipynb
- **Language**: Python with Scikit-learn, Plotly
- **Concepts**: NLP, tokenization, embeddings
- **Visualizations**: 3D scatter plots, interactive charts
- **Difficulty**: Intermediate

### WordsTokensUsingPytorch.ipynb
- **Language**: Python with PyTorch, Transformers
- **Concepts**: Deep learning, BERT, real tokenization
- **Visualizations**: Professional interactive plots
- **Difficulty**: Intermediate to Advanced

## Contributing

This is an educational repository. If you find errors or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face**: For the Transformers library and pre-trained models
- **PyTorch**: For the deep learning framework
- **Plotly**: For interactive visualizations
- **Scikit-learn**: For machine learning utilities
- **NumPy/Pandas**: For numerical computing and data manipulation

## AI Mastery Session 02 Source Code

This session introduces practical applications of Large Language Models (LLMs) with function calling capabilities and PDF processing for AI-powered document analysis.

### llm_chat_app.py - LLM Function Calling Application

**Description**: A comprehensive educational application demonstrating how to build chatbots that can execute custom functions using LLMs like Mistral AI and OpenAI's GPT models.

**What You'll Learn**:

**Core LLM Concepts**:
- **Function Calling**: How LLMs can execute custom Python functions based on user requests
- **API Integration**: Connecting to Mistral AI and OpenAI services with proper authentication
- **Streaming Responses**: Real-time text generation for interactive chat experiences
- **JSON Schema**: Defining function parameters and return types for LLM understanding

**Python Programming Skills**:
- **Advanced File Operations**: Pattern matching with glob, recursive directory traversal
- **Exception Handling**: Graceful error management in production applications
- **Type Hints**: Professional code documentation with Python's typing system
- **Dataclasses**: Modern Python data structure patterns
- **Command Line Interfaces**: Using argparse for professional CLI applications

**Software Engineering Patterns**:
- **API Client Design**: Building robust connections to external services
- **Configuration Management**: Handling API keys and environment variables securely
- **Modular Architecture**: Separating concerns with classes and methods
- **Error Recovery**: Handling network failures and API rate limits

**Function Calling Implementation**:
- **Tool Definition**: Creating JSON schemas that describe available functions
- **Parameter Validation**: Ensuring proper argument types and required fields
- **Result Processing**: Converting Python objects to LLM-readable formats
- **Context Management**: Maintaining conversation state and history

**Advanced Features**:
- **Enhanced Text Search**: Regular expressions, case sensitivity, whole-word matching
- **Context-Aware Results**: Showing surrounding lines for better understanding
- **File Type Filtering**: Searching specific file extensions efficiently
- **Result Limiting**: Preventing overwhelming output with smart pagination

**Usage Instructions**:
```bash
# Install required dependencies
pip install mistralai openai

# Set up API keys (choose one)
export MISTRAL_API_KEY="your-mistral-key"
export OPENAI_API_KEY="your-openai-key"

# Run with Mistral AI (default)
python llm_chat_app.py --provider mistral

# Run with OpenAI
python llm_chat_app.py --provider openai

# Enable streaming responses
python llm_chat_app.py --provider mistral --stream

# Specify working directory
python llm_chat_app.py --working-dir /path/to/your/project
```

**Example Interactions**:
```
You: List all Python files in this directory
Assistant: [Calls list_files function with pattern="*.py"]

You: Search for "import pandas" in my code
Assistant: [Calls search_files function to find pandas imports]

You: Create a README file for my project
Assistant: [Calls create_text_file function to generate documentation]
```

### PDF_Chat_Solution.ipynb - AI-Powered PDF Processing

**Description**: A complete Jupyter notebook demonstrating how to build an AI assistant that can process, analyze, and interact with PDF documents using advanced extraction techniques.

**What You'll Learn**:

**PDF Processing Mastery**:
- **Text Extraction**: Converting PDF content to searchable, analyzable text
- **Image Extraction**: Extracting embedded images with proper colorspace handling
- **Table Extraction**: Identifying and extracting tabular data from PDFs
- **Metadata Analysis**: Understanding PDF structure and properties

**Advanced Image Processing**:
- **CMYK to RGB Conversion**: Handling professional printing colorspaces
- **Multiple Extraction Methods**: Pixmap extraction and direct byte processing
- **Error Handling**: Graceful handling of corrupted or protected images
- **Format Optimization**: Converting to efficient formats for analysis

**AI Integration Patterns**:
- **Document Question Answering**: Ask questions about PDF content
- **Content Summarization**: Generate summaries of lengthy documents
- **Information Extraction**: Pull specific data points from documents
- **Multi-modal Processing**: Combining text and image analysis

**PyMuPDF (fitz) Library**:
- **Document Navigation**: Iterating through pages and elements
- **Object Extraction**: Accessing fonts, images, and annotations
- **Rendering Control**: Converting pages to images for analysis
- **Memory Management**: Efficient handling of large documents

**Mistral AI Integration**:
- **New API Format**: Using UserMessage, SystemMessage, AssistantMessage classes
- **Streaming Implementation**: Real-time response generation
- **Function Calling**: Integrating PDF operations as AI tools
- **Error Recovery**: Handling API failures gracefully

**Self-Contained Design**:
- **No External Dependencies**: All functions included in the notebook
- **Portable Solution**: Works without additional files or modules
- **Educational Structure**: Clear explanations for each code block
- **Debugging Capabilities**: Built-in troubleshooting and logging

**Usage Instructions**:
```bash
# Install required dependencies
pip install PyMuPDF mistralai pillow pandas

# Set up Mistral API key
export MISTRAL_API_KEY="your-mistral-key"

# Open Jupyter notebook
jupyter notebook PDF_Chat_Solution.ipynb

# Follow the notebook cells to:
# 1. Configure API settings
# 2. Upload or specify PDF file path
# 3. Run extraction functions
# 4. Start interactive chat with your PDF
```

**Key Features Demonstrated**:
- **Enhanced Image Extraction**: Solves macOS blank image issues with proper colorspace conversion
- **Interactive PDF Chat**: Ask questions about your PDF content
- **Comprehensive Extraction**: Text, images, tables, and metadata
- **Production-Ready Code**: Error handling, logging, and optimization
- **Educational Value**: Step-by-step explanations of each process

**Learning Outcomes**:
- Build production-ready PDF processing applications
- Integrate multiple AI services (Mistral, OpenAI) seamlessly
- Handle complex document formats and extraction challenges
- Create interactive AI applications with real-time capabilities
- Understand modern LLM integration patterns and best practices

**Prerequisites**:
- Python 3.8+ with virtual environment
- Jupyter Notebook or JupyterLab
- API access to Mistral AI or OpenAI
- Basic understanding of file operations and JSON

## Future Sessions

This series will continue with:
- **Session 3**: Advanced techniques with RAG and Model Context Protocol
- **Session 4**: Building no-code AI agents

---

**Happy Learning!** 🚀

*These notebooks provide a solid foundation for understanding modern AI systems. The mathematical principles you learn here are the same ones powering the most advanced AI models today.*