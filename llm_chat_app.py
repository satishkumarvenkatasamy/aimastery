#!/usr/bin/env python3
"""
LLM Chat Application with Function Calling - Educational Version
================================================================

This application demonstrates how to build a chatbot that can call custom functions
using Large Language Models (LLMs) like Mistral and OpenAI.

 WHAT IS FUNCTION CALLING?
Function calling (also known as "tool calling") is a feature that allows LLMs to:
1. Understand when they need to perform a specific action
2. Call predefined functions with appropriate parameters
3. Use the function results to provide better responses

 HOW IT WORKS:
1. You define functions that the LLM can call (like listing files, searching, etc.)
2. You describe these functions in a structured format (JSON schema)
3. The LLM analyzes user input and decides which function to call
4. Your code executes the function and returns results to the LLM
5. The LLM incorporates the results into its response

 LEARNING RESOURCES:
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Mistral Function Calling: https://docs.mistral.ai/capabilities/function_calling/
- JSON Schema: https://json-schema.org/understanding-json-schema/
- Python Type Hints: https://docs.python.org/3/library/typing.html

 EXAMPLE USAGE:
User: "List all Python files in this directory"
→ LLM calls list_files function with pattern="*.py"
→ Function returns list of Python files
→ LLM responds with formatted list

User: "Search for 'import pandas' in my code"
→ LLM calls search_files function with keyword="import pandas"
→ Function searches through files and returns matches
→ LLM responds with search results

Supports both Mistral and OpenAI models with file operation functions
"""

#  IMPORTS - Required libraries for the application
import os          # Operating system interface (file/directory operations)
import json        # JSON data handling (for function parameters)
import glob        # File pattern matching (for finding files with wildcards)
import argparse    # Command line argument parsing
from typing import List, Dict, Any, Optional  # Type hints for better code documentation
from dataclasses import dataclass             # Decorator for creating data classes

# LLM CLIENT IMPORTS
# We try to import both Mistral and OpenAI clients, but the app can work with just one
try:
    # Mistral AI client and message types
    # Learn more: https://docs.mistral.ai/getting-started/quickstart/
    from mistralai import Mistral, UserMessage, SystemMessage, AssistantMessage, ToolMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

try:
    # OpenAI client for GPT models
    # Learn more: https://platform.openai.com/docs/quickstart
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class Config:
    """
     CONFIGURATION CLASS
    
    This class stores all the settings for our LLM chat application.
    
     WHAT IS A DATACLASS?
    A dataclass is a Python decorator that automatically creates common methods
    like __init__, __repr__, etc. for classes that primarily store data.
    Learn more: https://docs.python.org/3/library/dataclasses.html
    
     CONFIGURATION OPTIONS:
    - model_provider: Which LLM service to use ("mistral" or "openai")
    - API keys: Authentication credentials for the services
    - Model names: Specific model versions to use
    - working_directory: Where file operations should happen
    """
    model_provider: str = "mistral"  # Default to Mistral AI
    mistral_api_key: Optional[str] = None    # Set this to use Mistral
    openai_api_key: Optional[str] = None     # Set this to use OpenAI
    mistral_model: str = "mistral-large-latest"  # Latest Mistral model
    openai_model: str = "gpt-3.5-turbo"         # OpenAI's ChatGPT model
    working_directory: str = "."                 # Current directory by default


class FileOperations:
    """
     FILE OPERATIONS CLASS
    
    This class contains all the functions that the LLM can call to perform file operations.
    Think of this as the "toolkit" that we give to the AI assistant.
    
     PURPOSE:
    When users ask questions like "what files are in this folder?" or "search for 'TODO' in my code",
    the LLM will call these functions to get the actual information from the file system.
    
     PYTHON CONCEPTS USED:
    - Classes: Grouping related functions together
    - Methods: Functions that belong to a class
    - File I/O: Reading from and writing to files
    - Exception handling: Dealing with errors gracefully
    
     Learn more about Python classes: https://docs.python.org/3/tutorial/classes.html
    """
    
    def __init__(self, working_dir: str = "."):
        """
        Initialize the FileOperations with a working directory.
        
        Args:
            working_dir: The directory where all file operations will happen
                        (default: "." means current directory)
        """
        self.working_dir = working_dir
    
    def list_files(self, pattern: str = "*") -> List[str]:
        """
         LIST FILES FUNCTION
        
        This function finds files that match a certain pattern (like "*.py" for Python files).
        
         WHAT IT DOES:
        - Searches the working directory for files matching the pattern
        - Returns a list of file paths
        
         PYTHON CONCEPTS:
        - glob.glob(): Pattern matching for filenames (like wildcards)
        - os.path.join(): Safely combines directory and file paths
        - List comprehension: [x for x in items if condition]
        
         EXAMPLES:
        - pattern="*" → finds all files
        - pattern="*.py" → finds all Python files
        - pattern="**/*.txt" → finds all .txt files recursively
        
         Learn about glob patterns: https://docs.python.org/3/library/glob.html
        
        Args:
            pattern: Glob pattern to match files (default: "*" means all files)
            
        Returns:
            List of file paths relative to the working directory
        """
        # Join the working directory with the pattern to get full search path
        search_pattern = os.path.join(self.working_dir, pattern)
        
        # Use glob to find all files matching the pattern
        files = glob.glob(search_pattern, recursive=True)
        
        # Filter to only include files (not directories) and make paths relative
        return [os.path.relpath(f, self.working_dir) for f in files if os.path.isfile(f)]
    
    def search_files(self, keyword: str, by_name: bool = True, by_content: bool = True) -> List[Dict[str, Any]]:
        """
         SEARCH FILES FUNCTION
        
        This function searches for a keyword in file names and/or file contents.
        It's like having a powerful search tool that can look through all your files.
        
         WHAT IT DOES:
        - Searches for a keyword in file names (like searching for "config" in filenames)
        - Searches for a keyword inside file contents (like searching for "TODO" in code)
        - Returns detailed information about matches found
        
         PYTHON CONCEPTS:
        - os.walk(): Recursively traverse directory tree
        - File I/O: Reading file contents with open()
        - Exception handling: try/except to handle errors gracefully
        - String methods: .lower(), .endswith(), .split()
        - Dictionaries: {"key": "value"} data structures
        
         EXAMPLES:
        - keyword="config" → finds files named "config.py" or containing "config"
        - keyword="import pandas" → finds files containing this import statement
        - keyword="TODO" → finds all TODO comments in your code
        
         Learn more:
        - File operations: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        - os.walk(): https://docs.python.org/3/library/os.html#os.walk
        
        Args:
            keyword: The text to search for (case-insensitive)
            by_name: Whether to search in file names (default: True)
            by_content: Whether to search inside file contents (default: True)
            
        Returns:
            List of dictionaries, each containing:
            - "file": relative path to the file
            - "matches": list of matches found (filename matches or content matches with line numbers)
        """
        results = []
        
        # Define which file types we can safely read as text
        text_extensions = ['.txt', '.py', '.md', '.json', '.csv', '.log', '.yml', '.yaml']
        all_files = []
        
        # Walk through all directories and subdirectories to find files
        for root, dirs, files in os.walk(self.working_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        # Check each file for matches
        for file_path in all_files:
            rel_path = os.path.relpath(file_path, self.working_dir)
            match_info = {"file": rel_path, "matches": []}
            
            # Search by filename
            if by_name and keyword.lower() in os.path.basename(file_path).lower():
                match_info["matches"].append({
                    "type": "filename", 
                    "content": os.path.basename(file_path)
                })
            
            # Search by content (only for text files to avoid binary files)
            if by_content and any(file_path.lower().endswith(ext) for ext in text_extensions):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if keyword.lower() in content.lower():
                            # Find specific lines containing the keyword
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if keyword.lower() in line.lower():
                                    match_info["matches"].append({
                                        "type": "content",
                                        "line_number": i + 1,
                                        "content": line.strip()
                                    })
                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read (binary files, permission issues)
                    continue
            
            # Only include files that have matches
            if match_info["matches"]:
                results.append(match_info)
        
        return results
    
    def create_text_file(self, filename: str, content: str) -> str:
        """
         CREATE TEXT FILE FUNCTION
        
        This function creates a new text file with the specified content.
        The LLM can use this to save information, create documentation, or generate code files.
        
         WHAT IT DOES:
        - Creates a new text file with the given name
        - Writes the provided content to the file
        - Creates any necessary parent directories
        - Returns a success message
        
         PYTHON CONCEPTS:
        - File I/O: Writing to files with open()
        - os.makedirs(): Creating directories if they don't exist
        - Context managers: "with" statement for safe file handling
        - String formatting: f-strings for creating messages
        
         EXAMPLES:
        - filename="notes.txt", content="My notes here" → creates a text file
        - filename="code/script.py", content="print('Hello')" → creates a Python file
        - filename="docs/readme.md", content="# My Project" → creates a markdown file
        
         Learn more:
        - File I/O: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        - Context managers: https://docs.python.org/3/reference/compound_stmts.html#with
        
        Args:
            filename: Name/path of the file to create (can include subdirectories)
            content: The text content to write to the file
            
        Returns:
            Success message with the full file path
        """
        # Create the full path by joining working directory with filename
        file_path = os.path.join(self.working_dir, filename)
        
        # Create parent directories if they don't exist (e.g., for "docs/readme.md")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the content to the file using UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"File created successfully: {file_path}"


class LLMChatApp:
    """
    MAIN LLM CHAT APPLICATION CLASS
    
    This is the heart of our application! It connects everything together:
    - The LLM (AI model) for understanding and generating text
    - The file operations for actually doing things with files
    - The function calling system that lets the AI use our tools
    
     WHAT THIS CLASS DOES:
    1. Initializes connection to the LLM service (Mistral or OpenAI)
    2. Manages the conversation with the user
    3. Decides when the LLM should call functions
    4. Executes the functions and returns results to the LLM
    5. Handles both regular and streaming responses
    
     KEY CONCEPTS:
    - API clients: Objects that connect to external services
    - Function calling: How LLMs can use external tools
    - Message handling: Managing conversation flow
    - Error handling: Graceful handling of problems
    
     Learn more:
    - API concepts: https://en.wikipedia.org/wiki/API
    - REST APIs: https://restfulapi.net/
    """
    
    def __init__(self, config: Config):
        """
        Initialize the LLM Chat Application
        
        This sets up everything needed to chat with the LLM and use functions.
        
        Args:
            config: Configuration object containing API keys and settings
        """
        self.config = config
        self.file_ops = FileOperations(config.working_directory)
        self.conversation_history = []  # Store conversation for context
        
        # Initialize LLM client based on chosen provider
        if config.model_provider == "mistral":
            if not MISTRAL_AVAILABLE:
                raise ImportError("Mistral client not available. Install with: pip install mistralai")
            self.client = Mistral(api_key=config.mistral_api_key)
        elif config.model_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI client not available. Install with: pip install openai")
            openai.api_key = config.openai_api_key
            self.client = openai
        else:
            raise ValueError(f"Unsupported model provider: {config.model_provider}")
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """
         FUNCTION DEFINITIONS FOR LLM
        
        This is THE MOST IMPORTANT METHOD for function calling!
        
         WHAT IT DOES:
        This method defines what functions the LLM can call and how to call them.
        It's like giving the AI a "manual" for using your tools.
        
         HOW IT WORKS:
        1. Returns a list of function descriptions in JSON Schema format
        2. Each function has a name, description, and parameter specification
        3. The LLM uses this information to decide when and how to call functions
        4. The parameters section tells the LLM what arguments are required/optional
        
         JSON SCHEMA FORMAT:
        Each function definition contains:
        - name: The function name to call
        - description: What the function does (helps LLM decide when to use it)
        - parameters: JSON Schema describing the function's arguments
        
         CRITICAL FOR BEGINNERS:
        - The "name" must exactly match your actual function name
        - The "description" helps the LLM understand WHEN to use the function
        - The "parameters" section defines the function's inputs
        - "required" array lists which parameters are mandatory
        
         Learn more:
        - JSON Schema: https://json-schema.org/understanding-json-schema/
        - OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
        - Mistral Function Calling: https://docs.mistral.ai/capabilities/function_calling/
        
        Returns:
            List of function definitions in JSON Schema format
        """
        return [
            {
                "name": "list_files",
                "description": "List files in the working directory matching a pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match files (default: '*')"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "search_files",
                "description": "Search for files by name or content containing a keyword",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "The keyword to search for"
                        },
                        "by_name": {
                            "type": "boolean",
                            "description": "Whether to search in file names (default: true)"
                        },
                        "by_content": {
                            "type": "boolean",
                            "description": "Whether to search in file contents (default: true)"
                        }
                    },
                    "required": ["keyword"]
                }
            },
            {
                "name": "create_text_file",
                "description": "Create a text file with LLM-generated content based on a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to create"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["filename", "content"]
                }
            }
        ]
    
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """
         FUNCTION EXECUTION METHOD
        
        This method actually runs the functions that the LLM wants to call.
        It's the bridge between the AI's decisions and your actual code.
        
         WHAT IT DOES:
        1. Receives function name and arguments from the LLM
        2. Maps the function name to the actual Python function
        3. Calls the function with the provided arguments
        4. Returns the result back to the LLM
        
         HOW THE PROCESS WORKS:
        LLM says: "I need to call list_files with pattern='*.py'"
        → This method receives: function_name="list_files", arguments={"pattern": "*.py"}
        → It calls: self.file_ops.list_files("*.py")
        → It returns: The list of Python files as JSON
        
         KEY CONCEPTS:
        - Function dispatch: Mapping names to actual functions
        - Argument unpacking: Converting dictionary to function parameters
        - Error handling: Graceful handling of function failures
        - JSON serialization: Converting Python objects to text for LLM
        
         IMPORTANT FOR BEGINNERS:
        - The function_name must match one of your defined functions exactly
        - Arguments come as a dictionary from the LLM
        - Results are returned as strings (LLMs work with text)
        - Always handle errors gracefully
        
        Args:
            function_name: Name of the function to call (e.g., "list_files")
            arguments: Dictionary of arguments to pass to the function
            
        Returns:
            String result from the function call (JSON formatted for complex data)
        """
        try:
            if function_name == "list_files":
                pattern = arguments.get("pattern", "*")
                result = self.file_ops.list_files(pattern)
                return json.dumps(result, indent=2)
            
            elif function_name == "search_files":
                keyword = arguments["keyword"]
                by_name = arguments.get("by_name", True)
                by_content = arguments.get("by_content", True)
                result = self.file_ops.search_files(keyword, by_name, by_content)
                return json.dumps(result, indent=2)
            
            elif function_name == "create_text_file":
                filename = arguments["filename"]
                content = arguments["content"]
                result = self.file_ops.create_text_file(filename, content)
                return result
            
            else:
                return f"Unknown function: {function_name}"
        
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    def chat_mistral(self, message: str) -> str:
        """
        MISTRAL CHAT METHOD
        
        This method handles communication with the Mistral AI service.
        It's where the magic of function calling happens!
        
         WHAT IT DOES:
        1. Sends user message to Mistral AI
        2. Includes function definitions so AI knows what tools are available
        3. Checks if AI wants to call any functions
        4. Executes functions and returns results
        5. Returns final response to user
        
         THE FUNCTION CALLING FLOW:
        User: "List all Python files"
        → Send to Mistral with available functions
        → Mistral responds: "I need to call list_files with pattern='*.py'"
        → We execute the function call
        → We return the results: ["script.py", "app.py"]
        → Mistral formats this into: "Here are the Python files: script.py, app.py"
        
         KEY CONCEPTS:
        - API calls: Sending HTTP requests to external services
        - Tool calling: New format where functions are "tools" the AI can use
        - Message format: UserMessage objects for new Mistral API
        - Response parsing: Extracting function calls from AI response
        
         IMPORTANT DETAILS:
        - tools=[{"type": "function", "function": func}] is the new format
        - tool_choice="auto" lets the AI decide whether to use functions
        - response.choices[0].message.tool_calls contains function calls
        - We parse JSON arguments and execute the actual functions
        
        Args:
            message: User's message/question
            
        Returns:
            AI's response (either direct text or function call results)
        """
        messages = [UserMessage(content=message)]
        
        response = self.client.chat.complete(
            model=self.config.mistral_model,
            messages=messages,
            tools=[{"type": "function", "function": func} for func in self.get_function_definitions()],
            tool_choice="auto"
        )
        
        if response.choices[0].message.tool_calls:
            # Handle function calls
            results = []
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                result = self.execute_function(function_name, arguments)
                results.append(f"Function {function_name} result: {result}")
            
            return "\n".join(results)
        else:
            return response.choices[0].message.content
    
    def chat_mistral_stream(self, message: str):
        """
         MISTRAL STREAMING CHAT METHOD
        
        This method provides real-time streaming responses from Mistral AI.
        Instead of waiting for the complete response, you get text as it's generated!
        
         WHAT IS STREAMING?
        Streaming means the AI sends its response piece by piece as it generates it,
        rather than waiting to send the complete response all at once.
        This creates a more interactive, chat-like experience.
        
         HOW STREAMING WORKS:
        1. Send request to Mistral with streaming enabled
        2. Receive response chunks (small pieces) as they're generated
        3. Yield each chunk immediately to the caller
        4. Handle function calls if they appear in the stream
        
         STREAMING CONCEPTS:
        - Generators: Python functions that yield values over time
        - Chunked responses: Data comes in small pieces
        - Real-time processing: Handle data as it arrives
        - chunk.data.choices[0].delta.content: New streaming format
        
         STREAMING SPECIFICS:
        - Uses client.chat.stream() instead of client.chat.complete()
        - Yields content chunks as they arrive
        - Function calls can also appear in streaming format
        - Great for long responses or interactive feel
        
         Learn more:
        - Python generators: https://docs.python.org/3/tutorial/classes.html#generators
        - Streaming APIs: https://en.wikipedia.org/wiki/Streaming_API
        
        Args:
            message: User's message/question
            
        Yields:
            Text chunks as they're generated by the AI
        """
        messages = [UserMessage(content=message)]
        
        stream = self.client.chat.stream(
            model=self.config.mistral_model,
            messages=messages,
            tools=[{"type": "function", "function": func} for func in self.get_function_definitions()],
            tool_choice="auto"
        )
        
        for chunk in stream:
            if chunk.data.choices[0].delta.content is not None:
                yield chunk.data.choices[0].delta.content
            elif chunk.data.choices[0].delta.tool_calls:
                # Handle function calls in streaming
                for tool_call in chunk.data.choices[0].delta.tool_calls:
                    if tool_call.function.name and tool_call.function.arguments:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        result = self.execute_function(function_name, arguments)
                        yield f"\nFunction {function_name} result: {result}\n"
    
    def chat_openai(self, message: str) -> str:
        """Chat with OpenAI model"""
        messages = [{"role": "user", "content": message}]
        
        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            tools=[{"type": "function", "function": func} for func in self.get_function_definitions()],
            tool_choice="auto"
        )
        
        if response.choices[0].message.tool_calls:
            # Handle function calls
            results = []
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                result = self.execute_function(function_name, arguments)
                results.append(f"Function {function_name} result: {result}")
            
            return "\n".join(results)
        else:
            return response.choices[0].message.content
    
    def chat(self, message: str) -> str:
        """Main chat method that routes to appropriate LLM"""
        if self.config.model_provider == "mistral":
            return self.chat_mistral(message)
        elif self.config.model_provider == "openai":
            return self.chat_openai(message)
        else:
            return "Unsupported model provider"
    
    def chat_stream(self, message: str):
        """Main streaming chat method that routes to appropriate LLM"""
        if self.config.model_provider == "mistral":
            for chunk in self.chat_mistral_stream(message):
                yield chunk
        elif self.config.model_provider == "openai":
            # OpenAI streaming would be implemented here
            yield self.chat_openai(message)
        else:
            yield "Unsupported model provider"


def main():
    """
     MAIN FUNCTION - COMMAND LINE INTERFACE
    
    This is the entry point of the application when run from the command line.
    It handles user input, sets up the application, and runs the chat loop.
    
     WHAT IT DOES:
    1. Parses command line arguments (--provider, --stream, etc.)
    2. Sets up API keys from environment variables or command line
    3. Creates the configuration and initializes the chat app
    4. Runs the interactive chat loop
    5. Handles user input and displays AI responses
    
     KEY CONCEPTS:
    - Command line parsing: Using argparse to handle CLI arguments
    - Environment variables: Reading settings from system environment
    - Interactive loops: Continuous user interaction until quit
    - Exception handling: Graceful handling of errors
    
     EXAMPLE USAGE:
    python llm_chat_app.py --provider mistral --stream
    python llm_chat_app.py --provider openai --working-dir /path/to/files
    
     IMPORTANT FOR BEGINNERS:
    - if __name__ == "__main__": ensures this only runs when script is executed directly
    - The while True loop keeps the chat running until user types 'quit'
    - try/except blocks handle errors gracefully
    - Input/output handling creates the conversational interface
    
     Learn more:
    - Command line arguments: https://docs.python.org/3/library/argparse.html
    - Environment variables: https://docs.python.org/3/library/os.html#os.environ
    """
    parser = argparse.ArgumentParser(description="LLM Chat Application with File Operations")
    parser.add_argument("--provider", choices=["mistral", "openai"], default="mistral",
                       help="LLM provider to use")
    parser.add_argument("--working-dir", default=".", help="Working directory for file operations")
    parser.add_argument("--mistral-key", help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming responses (Mistral only)")
    
    args = parser.parse_args()
    
    # Configure API keys
    mistral_key = args.mistral_key or os.getenv("MISTRAL_API_KEY")
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    
    if args.provider == "mistral" and not mistral_key:
        print("Error: Mistral API key required. Set MISTRAL_API_KEY env var or use --mistral-key")
        return
    
    if args.provider == "openai" and not openai_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --openai-key")
        return
    
    # Create configuration
    config = Config(
        model_provider=args.provider,
        mistral_api_key=mistral_key,
        openai_api_key=openai_key,
        working_directory=args.working_dir
    )
    
    # Initialize chat app
    try:
        app = LLMChatApp(config)
    except Exception as e:
        print(f"Error initializing chat app: {e}")
        return
    
    print(f"LLM Chat App initialized with {args.provider.upper()} provider")
    print(f" Working directory: {os.path.abspath(args.working_dir)}")
    if args.stream and args.provider == "mistral":
        print(" Streaming mode enabled")
    print("\nAvailable functions:")
    print("- list_files: List files matching a pattern")
    print("- search_files: Search files by name or content")
    print("- create_text_file: Create a text file with LLM-generated content")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("Assistant: ", end="")
            if args.stream and args.provider == "mistral":
                for chunk in app.chat_stream(user_input):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = app.chat(user_input)
                print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    """
     PROGRAM ENTRY POINT
    
    This special Python construct means: "Only run main() if this file is executed directly"
    
    Why is this important?
    - If you run: python llm_chat_app.py → main() will execute
    - If you import: from llm_chat_app import LLMChatApp → main() will NOT execute
    - This allows the file to be both a script and a module
    
     Learn more: https://docs.python.org/3/library/__main__.html
    """
    main()

#  CONGRATULATIONS! 
# You've just explored a complete LLM Function Calling application!
# 
#  WHAT YOU'VE LEARNED:
# - How LLMs can call custom functions
# - JSON Schema for function definitions
# - API clients for Mistral and OpenAI
# - Streaming responses for real-time interaction
# - File operations with Python
# - Error handling and user input
# - Command line interface development
# 
#  NEXT STEPS:
# 1. Try running the application with different providers
# 2. Add your own custom functions
# 3. Experiment with different function descriptions
# 4. Add more complex file operations
# 5. Implement conversation history
# 6. Add support for other LLM providers
# 
#  FURTHER LEARNING:
# - Build more complex functions (database operations, web scraping, etc.)
# - Learn about prompt engineering for better function calling
# - Explore advanced LLM features like multi-turn conversations
# - Study API design patterns and best practices
# 
# Happy coding! Looking forward for interesting Hackathon ideas!
