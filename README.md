# HALTGPT - Advanced OpenAI Chat Interface

HALTGPT is a feature-rich desktop application that provides an enhanced interface for interacting with OpenAI's language models. It offers multiple chat tabs, agent-based interactions, custom instructions, and various tools to improve your AI experience.

![image](https://github.com/user-attachments/assets/16423254-8e60-4645-8ba6-534ea510d033)
![image](https://github.com/user-attachments/assets/9301cac2-ad5c-4432-bfb2-f435aa4ef514)


## Features

- **Multiple Chat Pages**: Maintain separate conversations in up to 20 tabs
- **Model Selection**: Choose from various OpenAI models including GPT-4o, GPT-4-turbo, and more
- **Agent Mode**: Enable single agent or multi-agent dialogs for specialized interactions
- **Custom Instructions**: Set system and developer instructions for consistent AI behavior
- **Image Generation**: Generate images with DALL-E 3 and DALL-E 2 models
- **Customizable UI**: Choose from multiple themes or create your own
- **Custom Actions**: Create and manage reusable prompt templates and actions
- **Session Management**: Save and load your conversation sessions
- **Streaming Responses**: See AI responses as they're generated in real-time

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone this repository or download the source code
```
git clone https://github.com/yourusername/HALTGPT.git
cd HALTGPT
```

2. Create a virtual environment
```
python -m venv .venv
```

3. Activate the virtual environment
   - Windows: `.venv\Scripts\activate`
   - Linux/MacOS: `source .venv/bin/activate`

4. Install the required packages
```
pip install -r requirements.txt
```

5. Create a `.env` file in the project root directory with your OpenAI API key
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Launch the application
```
python .venv/HALTGPT.py
```

2. The interface has the following main components:
   - **Left panel**: Chat tabs with input and output areas
   - **Right panel**: Settings, instructions, appearance, tools, and agent configurations

### Basic Chat

1. Enter your prompt in the input area of any chat tab
2. Click "Generate" or press Ctrl+Enter to get a response
3. View the response in the output area

### Agent Mode

1. Go to the "Agents" tab in the right panel
2. Enable "Agent Mode" with the checkbox
3. Configure agent name and instructions
4. Use the chat as normal, but now with agent capabilities

### Multi-Agent Dialog

1. Go to the "Agents" tab
2. Enable both "Agent Mode" and "Multi-Agent Dialog"
3. Configure the number of agents and their roles
4. Use the chat to generate conversations between the agents

### Image Generation

1. Go to the "Tools" tab
2. Enter a prompt in the "Image Generation" section
3. Select the model (DALL-E 3 or DALL-E 2) and size
4. Click "Generate Image" to create an image
5. Use "Save Image" to save the generated image

### Custom Actions

1. Click "Manage Actions" in any chat tab
2. Create custom actions to insert text templates or execute operations
3. Use these actions from the dropdown menu in chat tabs

## Keyboard Shortcuts

- **Ctrl+Enter**: Generate response for current tab
- **Ctrl+Tab**: Switch to next tab
- **Ctrl+Shift+Tab**: Switch to previous tab
- **Ctrl+S**: Save output to file
- **Ctrl+L**: Clear current chat history
- **Escape**: Stop generation

## Settings and Customization

### Model Settings

Configure temperature, top_p, and max tokens in the "Chat" tab of the settings panel.

### Instructions

Set system and developer instructions in the "Instructions" tab to guide the AI's behavior across all chats.

### Appearance

Customize the application's look and feel in the "Appearance" tab:
- Choose from Light, Dark, and Blue themes
- Adjust font size and family
- Create custom themes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for providing the API
- PyQt5 for the UI framework
- All contributors who have helped to enhance this project
