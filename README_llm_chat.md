# LLM Chat Application with File Operations

A simple Python chat application that demonstrates LLM function calling capabilities with file operations. Supports both Mistral and OpenAI models.

## Features

- **File Operations**: List, search, and create files using LLM function calls
- **Dual Model Support**: Works with both Mistral and OpenAI models
- **Free API Compatible**: Designed to work with free API tiers
- **Local File Access**: Operates on files in the developer's local machine

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
```bash
# For Mistral
export MISTRAL_API_KEY="your_mistral_api_key_here"

# For OpenAI
export OPENAI_API_KEY="your_openai_api_key_here"
```

## Usage

### Basic Usage

```bash
# Using Mistral (default)
python llm_chat_app.py

# Using OpenAI
python llm_chat_app.py --provider openai

# Specify working directory
python llm_chat_app.py --working-dir /path/to/your/files
```

### Available Functions

The LLM can call these functions:

1. **`list_files`**: List files matching a pattern
2. **`search_files`**: Search files by name or content
3. **`create_text_file`**: Create a text file with LLM-generated content

### Example Conversations

#### Listing Files
```
You: List all Python files in the current directory
Assistant: Function list_files result: [
  "llm_chat_app.py",
  "test_script.py"
]
```

#### Searching Files
```
You: Search for files containing the word "function"
Assistant: Function search_files result: [
  {
    "file": "llm_chat_app.py",
    "matches": [
      {
        "type": "content",
        "line_number": 45,
        "content": "def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:"
      }
    ]
  }
]
```

#### Creating Files
```
You: Create a file called "hello.txt" with a greeting message
Assistant: Function create_text_file result: File created successfully: ./hello.txt
```

## Command Line Options

- `--provider`: Choose between "mistral" or "openai" (default: mistral)
- `--working-dir`: Set the working directory for file operations (default: current directory)
- `--mistral-key`: Specify Mistral API key directly
- `--openai-key`: Specify OpenAI API key directly

## API Keys

You can obtain free API keys from:
- **Mistral**: https://console.mistral.ai/
- **OpenAI**: https://platform.openai.com/

Both providers offer free tiers suitable for testing and light usage.

## Example Session

```bash
$ python llm_chat_app.py --provider mistral
🤖 LLM Chat App initialized with MISTRAL provider
📁 Working directory: /Users/user/myproject

Available functions:
- list_files: List files matching a pattern
- search_files: Search files by name or content
- create_text_file: Create a text file with LLM-generated content

Type 'quit' to exit

You: What Python files do I have?
Assistant: Function list_files result: [
  "app.py",
  "utils.py",
  "test.py"
]

You: Create a simple TODO list file
Assistant: Function create_text_file result: File created successfully: ./todo.txt

You: quit
Goodbye!
```

## Architecture

The application consists of:

- **`Config`**: Configuration dataclass for API keys and settings
- **`FileOperations`**: Core file operation functions
- **`LLMChatApp`**: Main chat application with function calling logic
- **CLI Interface**: Command-line interface for user interaction

## Notes

- The application operates on the local file system where it's running
- Text file search supports common extensions (.txt, .py, .md, .json, etc.)
- Function calling follows the OpenAI function calling format
- Both Mistral and OpenAI implementations use the same function definitions