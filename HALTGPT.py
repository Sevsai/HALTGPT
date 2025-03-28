#!/usr/bin/env python
import os
import sys
import time
import json
import logging
import io
import requests
import base64
import asyncio
import openai
import threading

# Global event for stopping operations
stop_event = threading.Event()
from threading import Thread, Event
import asyncio

from PyQt5.QtWidgets import (QApplication, QMainWindow, QSplitter, QTabWidget, 
                             QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QTextEdit, QFrame, QCheckBox, QComboBox, QSpinBox, 
                             QScrollArea, QLineEdit, QFileDialog, QMessageBox, QColorDialog, QListWidget,
                             QShortcut, QToolButton, )
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer, QObject, QMimeData, QByteArray
from PyQt5.QtGui import QFont, QColor, QPixmap, QTextCursor, QIcon, QDrag, QKeySequence
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

from PIL import Image, ImageQt
from dotenv import load_dotenv

# Replace Gemini imports with OpenAI
import openai
from openai import OpenAI

# ------------------------------------------------------------------------------
# Logging Setup
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ------------------------------------------------------------------------------
# Global Variables and Initial Data Loading
openai_client = None
try:
    # Load API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
    else:
        logging.error("OPENAI_API_KEY environment variable not set")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")

# Update model settings to use OpenAI's models
model_settings = {
    "model": "gpt-4o",
    "max_tokens": 4096,  # Increased from default 1024 to allow longer responses
}
stop_event = Event()

# Add these constants after the model_settings definition
APP_SETTINGS_FILE = "app_settings.json"
CONVERSATION_LOG_FILE = "conversation_log.txt"

# Add helper functions after model_settings definition
def is_o_family_model(model_name):
    """Check if model is part of the O family (GPT-4o, o-mini, o1, o3, etc.)"""
    model_name = model_name.lower()
    return "o" in model_name or "o1" in model_name or "o3" in model_name

def supports_top_p(model_name):
    """Check if model supports top_p parameter"""
    return not is_o_family_model(model_name)

def supports_system_instructions(model_name):
    """Check if model supports system instructions"""
    model_name = model_name.lower()
    # All current OpenAI models support system instructions except very old ones
    return True

def supports_developer_instructions(model_name):
    """Check if model should use developer instructions"""
    return is_o_family_model(model_name)

def supports_temperature(model_name):
    """Check if model supports temperature parameter"""
    model_name = model_name.lower()
    return not ("o1" in model_name or "o3" in model_name)

def supports_max_tokens(model_name):
    """Check which token limit parameter the model supports"""
    model_name = model_name.lower()
    if "o1" in model_name or "o3" in model_name:
        return "max_completion_tokens"
    return "max_tokens"

# Add these helper functions to handle settings and logging
def save_app_settings(app_instance):
    """Save all application settings to a single JSON file"""
    settings = {
        "system_instructions": app_instance.system_instructions_text.toPlainText() if hasattr(app_instance, 'system_instructions_text') else "",
        "developer_instructions": app_instance.developer_instructions_text.toPlainText() if hasattr(app_instance, 'developer_instructions_text') else "",
        "custom_actions": app_instance.button_functions,
        "model_settings": model_settings,
        "theme": current_theme,
        "agent_settings": {
            "agent_enabled": app_instance.agent_enabled,
            "multi_agent_enabled": app_instance.multi_agent_enabled,
            "agent_roles": app_instance.agent_roles
        }
    }
    
    try:
        with open(APP_SETTINGS_FILE, "w", encoding="utf-8") as file:
            json.dump(settings, file, indent=4)
        return True
    except Exception as e:
        logging.error(f"Failed to save application settings: {e}")
        return False

def load_app_settings(app_instance):
    """Load all application settings from a single JSON file"""
    global current_theme, model_settings
    
    try:
        if not os.path.exists(APP_SETTINGS_FILE):
            logging.info(f"Settings file {APP_SETTINGS_FILE} not found. Using defaults.")
            return False
            
        with open(APP_SETTINGS_FILE, "r", encoding="utf-8") as file:
            settings = json.load(file)
            
            # Load system and developer instructions
            if hasattr(app_instance, 'system_instructions_text') and "system_instructions" in settings:
                app_instance.system_instructions_text.setPlainText(settings["system_instructions"])
                
            if hasattr(app_instance, 'developer_instructions_text') and "developer_instructions" in settings:
                app_instance.developer_instructions_text.setPlainText(settings["developer_instructions"])
            
            # Load custom actions
            if "custom_actions" in settings:
                app_instance.button_functions = settings["custom_actions"]
                # Update action menus
                for actions_menu in app_instance.actions_menus:
                    actions_menu.clear()
                    actions_menu.addItem("Actions")
                    for action_name in app_instance.button_functions.keys():
                        actions_menu.addItem(action_name)
            
            # Load model settings
            if "model_settings" in settings:
                model_settings.update(settings["model_settings"])
                # Update UI to reflect loaded settings
                if hasattr(app_instance, 'model_selector'):
                    index = app_instance.model_selector.findText(model_settings["model"])
                    if index >= 0:
                        app_instance.model_selector.setCurrentIndex(index)
                
                if hasattr(app_instance, 'temperature_entry'):
                    app_instance.temperature_entry.setText(str(model_settings.get("temperature", 0.7)))
                
                if hasattr(app_instance, 'top_p_entry'):
                    app_instance.top_p_entry.setText(str(model_settings.get("top_p", 1.0)))
                
                if hasattr(app_instance, 'max_tokens_entry'):
                    app_instance.max_tokens_entry.setText(str(model_settings.get("max_tokens", 4096)))
                    
                if hasattr(app_instance, 'show_thinking_checkbox'):
                    app_instance.show_thinking_checkbox.setChecked(model_settings.get("show_thinking", False))
            
            # Load theme
            if "theme" in settings:
                app_instance.apply_theme(settings["theme"])
            
            # Load agent settings
            if "agent_settings" in settings:
                agent_settings = settings["agent_settings"]
                if "agent_enabled" in agent_settings:
                    app_instance.agent_enabled = agent_settings["agent_enabled"]
                    if hasattr(app_instance, 'agent_toggle'):
                        app_instance.agent_toggle.setChecked(app_instance.agent_enabled)
                
                if "multi_agent_enabled" in agent_settings:
                    app_instance.multi_agent_enabled = agent_settings["multi_agent_enabled"]
                    if hasattr(app_instance, 'multi_agent_toggle'):
                        app_instance.multi_agent_toggle.setChecked(app_instance.multi_agent_enabled)
                
                if "agent_roles" in agent_settings:
                    app_instance.agent_roles = agent_settings["agent_roles"]
                    app_instance.update_agent_roles_ui()
            
        return True
    except Exception as e:
        logging.error(f"Failed to load application settings: {e}")
        return False

def log_conversation(page_index, role, content):
    """Log conversation to text file with timestamp"""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(CONVERSATION_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] Page {page_index+1} - {role}: {content}\n\n")
    except Exception as e:
        logging.error(f"Failed to log conversation: {e}")

# Add a worker class for thread-safe OpenAI API calls
class OpenAIWorker(QObject):
    generation_complete = pyqtSignal(str)
    generation_error = pyqtSignal(str)
    
    def __init__(self, prompt, model_settings):
        super().__init__()
        self.prompt = prompt
        self.model_settings = model_settings
    
    @pyqtSlot()
    def generate(self):
        try:
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                self.generation_error.emit("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                return
            
            # Initialize the OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Get the model name
            model_name = self.model_settings["model"]
            
            # Configure generation settings based on model capabilities
            generation_config = {
                "model": model_name,
            }
            
            # Add temperature if supported
            if supports_temperature(model_name):
                generation_config["temperature"] = float(self.model_settings.get("temperature", 0.7))
            
            # Add appropriate token limit parameter based on model
            token_limit = int(self.model_settings.get("max_tokens", 4096))
            token_param = supports_max_tokens(model_name)
            generation_config[token_param] = token_limit
            
            # Add top_p only for models that support it
            if supports_top_p(model_name) and "top_p" in self.model_settings:
                generation_config["top_p"] = float(self.model_settings.get("top_p", 1.0))
            
            # Parse the prompt to get messages
            try:
                # Try to parse as JSON if it's in messages format
                messages = json.loads(self.prompt)
            except json.JSONDecodeError:
                # If not JSON, treat as a plain text prompt
                messages = [{"role": "user", "content": self.prompt}]
            
            # Process any system or developer instructions
            if isinstance(messages, list) and messages:
                # Check if we need to add system instructions
                has_system_message = any(msg.get("role") == "system" for msg in messages)
                
                # If no system message and the model supports it, try to add one
                if not has_system_message and hasattr(self, 'system_instructions') and self.system_instructions and supports_system_instructions(model_name):
                    messages.insert(0, {"role": "system", "content": self.system_instructions})
                
                # For o-family models, if we have developer instructions, append them
                if supports_developer_instructions(model_name) and hasattr(self, 'developer_instructions') and self.developer_instructions:
                    dev_msg = {"role": "system", "content": f"Developer instructions: {self.developer_instructions}"}
                    messages.insert(1 if has_system_message else 0, dev_msg)
            
            # Add thinking process config if enabled
            if self.model_settings.get("show_thinking", False):
                # Find or create a system message for thinking instructions
                thinking_prompt = "Please think step by step and show your reasoning process."
                
                if isinstance(messages, list):
                    has_system = any(msg.get("role") == "system" for msg in messages)
                    if has_system:
                        # Append to existing system message
                        for msg in messages:
                            if msg.get("role") == "system":
                                msg["content"] += f"\n\n{thinking_prompt}"
                                break
                    else:
                        # Add new system message
                        messages.insert(0, {"role": "system", "content": thinking_prompt})
            
            # Add messages to config
            generation_config["messages"] = messages
            
            # Generate content with OpenAI
            response = client.chat.completions.create(**generation_config)
            
            # Extract text from response
            if response and response.choices and len(response.choices) > 0:
                output_text = response.choices[0].message.content
            else:
                output_text = "No response generated."
            
            # Emit the result signal
            self.generation_complete.emit(output_text)
            
        except Exception as e:
            logging.error(f"Failed to generate response: {e}")
            self.generation_error.emit(str(e))

# Add a worker class for thread-safe image generation using DALL-E
class ImageGenerationWorker(QObject):
    generation_complete = pyqtSignal(object)  # Will pass the PIL Image object
    generation_error = pyqtSignal(str)
    
    def __init__(self, prompt, model_name, width, height):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
        self.width = width
        self.height = height
    
    @pyqtSlot()
    def generate(self):
        try:
            generated_image = None
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.generation_error.emit("API key not found. Set OPENAI_API_KEY environment variable.")
                return
            
            # Initialize client with API key
            client = OpenAI(api_key=api_key)
            
            # Generate image based on selected model
            if self.model_name == "DALL-E 3":
                # Map dimensions to closest supported size for DALL-E 3
                size = self.map_to_dalle_size(self.width, self.height)
                
                # Generate image with DALL-E 3
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=self.prompt,
                    size=size,
                    quality="standard",
                    n=1
                )
                
                # Get the image URL and download it
                if hasattr(response, 'data') and len(response.data) > 0:
                    image_url = response.data[0].url
                    image_response = requests.get(image_url)
                    generated_image = Image.open(io.BytesIO(image_response.content))
                
            elif self.model_name == "DALL-E 2":
                # Map dimensions to closest supported size for DALL-E 2
                size = self.map_to_dalle_size(self.width, self.height, model="dall-e-2")
                
                # Generate image with DALL-E 2
                response = client.images.generate(
                    model="dall-e-2",
                    prompt=self.prompt,
                    size=size,
                    n=1
                )
                
                # Get the image URL and download it
                if hasattr(response, 'data') and len(response.data) > 0:
                    image_url = response.data[0].url
                    image_response = requests.get(image_url)
                    generated_image = Image.open(io.BytesIO(image_response.content))
            
            # Emit the result signal with the generated image
            if generated_image:
                self.generation_complete.emit(generated_image)
            else:
                self.generation_error.emit("Failed to generate image")
                
        except Exception as e:
            logging.error(f"Image generation failed: {e}")
            self.generation_error.emit(str(e))
    
    def map_to_dalle_size(self, width, height, model="dall-e-3"):
        """Map requested dimensions to the closest supported size for DALL-E models"""
        if model == "dall-e-3":
            # DALL-E 3 supports 1024x1024, 1024x1792, 1792x1024
            if width == height:
                return "1024x1024"
            elif width > height:
                return "1792x1024"
            else:
                return "1024x1792"
        else:  # DALL-E 2
            # DALL-E 2 supports 256x256, 512x512, 1024x1024
            if width <= 256 and height <= 256:
                return "256x256"
            elif width <= 512 and height <= 512:
                return "512x512"
            else:
                return "1024x1024"

# Add this class after the OpenAIWorker class
class OpenAIStreamingWorker(QObject):
    token_received = pyqtSignal(str)
    generation_complete = pyqtSignal()
    generation_error = pyqtSignal(str)
    
    def __init__(self, prompt, model_settings):
        super().__init__()
        self.prompt = prompt
        self.model_settings = model_settings
    
    @pyqtSlot()
    def generate(self):
        try:
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                self.generation_error.emit("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                return
            
            # Initialize the OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Get the model name
            model_name = self.model_settings["model"]
            
            # Configure generation settings based on model capabilities
            generation_config = {
                "model": model_name,
                "stream": True  # Enable streaming
            }
            
            # Add temperature if supported
            if supports_temperature(model_name):
                generation_config["temperature"] = float(self.model_settings.get("temperature", 0.7))
            
            # Add appropriate token limit parameter based on model
            token_limit = int(self.model_settings.get("max_tokens", 4096))
            token_param = supports_max_tokens(model_name)
            generation_config[token_param] = token_limit
            
            # Add top_p only for models that support it
            if supports_top_p(model_name) and "top_p" in self.model_settings:
                generation_config["top_p"] = float(self.model_settings.get("top_p", 1.0))
            
            # Parse the prompt to get messages
            try:
                # Try to parse as JSON if it's in messages format
                messages = json.loads(self.prompt)
            except json.JSONDecodeError:
                # If not JSON, treat as a plain text prompt
                messages = [{"role": "user", "content": self.prompt}]
            
            # Add messages to config
            generation_config["messages"] = messages
            
            # Generate streaming content with OpenAI
            stream = client.chat.completions.create(**generation_config)
            
            # Process the stream
            full_response = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        # Emit each token as it arrives
                        self.token_received.emit(delta.content)
                        full_response += delta.content
            
            # Signal completion
            self.generation_complete.emit()
            
        except Exception as e:
            logging.error(f"Failed to generate streaming response: {e}")
            self.generation_error.emit(str(e))

# Add a streaming worker class for agent responses
class AgentStreamingWorker(QObject):
    token_received = pyqtSignal(str)
    generation_complete = pyqtSignal()
    generation_error = pyqtSignal(str)
    
    def __init__(self, prompt, agent_name, agent_instructions, agent_model, agent_memory_content=""):
        super().__init__()
        self.prompt = prompt
        self.agent_name = agent_name
        self.agent_instructions = agent_instructions
        self.agent_model = agent_model
        self.agent_memory_content = agent_memory_content
    
    @pyqtSlot()
    def generate(self):
        try:
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                self.generation_error.emit("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                return
            
            # Initialize the OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Build the messages array for OpenAI
            messages = [
                {"role": "system", "content": f"You are {self.agent_name}. {self.agent_instructions}{self.agent_memory_content}"},
                {"role": "user", "content": self.prompt}
            ]
            
            # Configure generation settings based on model capabilities
            generation_config = {
                "model": self.agent_model,
                "stream": True,  # Enable streaming
                "messages": messages
            }
            
            # Add temperature parameter only for models that support it
            if not ("o1" in self.agent_model.lower() or "o3" in self.agent_model.lower()):
                generation_config["temperature"] = 0.7
                generation_config["top_p"] = 1.0
            
            # Add appropriate token limit parameter based on model
            token_limit = 20000  # Limit token length for faster responses
            if "o1" in self.agent_model.lower() or "o3" in self.agent_model.lower():
                generation_config["max_completion_tokens"] = token_limit
            else:
                generation_config["max_tokens"] = token_limit
            
            # Generate streaming content with OpenAI
            stream = client.chat.completions.create(**generation_config)
            
            # Process the stream
            full_response = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        # Emit each token as it arrives
                        self.token_received.emit(delta.content)
                        full_response += delta.content
            
            # Signal completion
            self.generation_complete.emit()
            
        except Exception as e:
            logging.error(f"Failed to generate agent streaming response: {e}")
            self.generation_error.emit(str(e))

# Dictionary to store conversation history for each chat page
chat_histories = {}  # {page_index: [{"role": "user/model", "content": "message"}]}

# Themes - Using QSS for styling
themes = {
    "Dark": {
        "bg": "#2c3e50",
        "fg": "#ecf0f1",
        "input_bg": "#34495e",
        "output_bg": "#34495e",
        "accent": "#2980b9",
        "border": "#3d566e"
    },
    "Light": {
        "bg": "#f0f0f0",
        "fg": "#2c3e50",
        "input_bg": "#ffffff",
        "output_bg": "#ffffff",
        "accent": "#007AFF",
        "border": "#cccccc"
    },
    "Blue": {
        "bg": "#1e3d59",
        "fg": "#ecf0f1",
        "input_bg": "#3a506b",
        "output_bg": "#3a506b",
        "accent": "#0055D4",
        "border": "#4a6fa5"
    }
}
current_theme = "Light"

# Default fonts used for UI text
current_fonts = {
    "label": QFont("Segoe UI", 12, QFont.Bold),
    "input": QFont("Segoe UI", 11),
    "output": QFont("Segoe UI", 11),
    "heading": QFont("Segoe UI", 18, QFont.Bold),
    "button": QFont("Segoe UI", 11)
}

# ------------------------------------------------------------------------------
# Toast Notification Widget
class Toast(QWidget):
    """Displays temporary toast notifications"""
    
    def __init__(self, parent, text, duration=3000, background=None, foreground=None):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # Get theme colors if not specified
        theme = themes[current_theme]
        self.background = background or theme.get("accent", "#007AFF")
        self.foreground = foreground or theme.get("fg", "#FFFFFF")
        self.duration = duration
        
        # Position at bottom center of parent
        parent_rect = parent.geometry()
        self.setGeometry(
            parent_rect.center().x() - 150, 
            parent_rect.bottom() - 100,
            300, 80
        )
        
        # Layout
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        
        # Style
        self.setStyleSheet(f"""
            background-color: {self.background};
            color: {self.foreground};
            border-radius: 10px;
            padding: 15px;
        """)
        
        # Message
        self.label = QLabel(text, self)
        self.label.setStyleSheet(f"color: {self.foreground}; font-size: 11pt;")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        
        # Animation
        self.setWindowOpacity(0.0)
        self.fade_in()
        
        # Timer for auto-close
        if self.duration > 0:
            QTimer.singleShot(self.duration, self.fade_out)
            
    def fade_in(self):
        # Call parent QWidget's show method, not our static method
        super().show()  # Use super() instead of self.show()
        for i in range(1, 11):
            alpha = i / 10
            self.setWindowOpacity(alpha)
            QApplication.processEvents()
            time.sleep(0.02)
    
    def fade_out(self):
        for i in range(10, -1, -1):
            alpha = i / 10
            self.setWindowOpacity(alpha)
            QApplication.processEvents()
            time.sleep(0.02)
        self.hide()
        self.deleteLater()
    
    @staticmethod
    def show(parent, text, duration=3000, background=None, foreground=None):
        """Static method to quickly show a toast"""
        return Toast(parent, text, duration, background, foreground)

# ------------------------------------------------------------------------------
# Agent Role Widget
class AgentRoleWidget(QFrame):
    """A draggable widget for agent roles in the multi-agent setup"""
    
    def __init__(self, parent, agent_index, role_text, on_text_changed=None, on_role_moved=None):
        super().__init__(parent)
        self.agent_index = agent_index
        self.role_text = role_text
        self.on_text_changed = on_text_changed
        self.on_role_moved = on_role_moved
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setLineWidth(2)
        
        # Make the widget accept drops
        self.setAcceptDrops(True)
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header with agent number and drag handle
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Agent label with tooltip
        self.agent_label = QLabel(f"Agent {self.agent_index + 1}:")
        self.agent_label.setFont(current_fonts["label"])
        self.agent_label.setToolTip(f"Role for Agent {self.agent_index + 1} - Drag to reorder")
        header_layout.addWidget(self.agent_label)
        
        # Role template dropdown
        role_selector_frame = QFrame()
        role_selector_layout = QHBoxLayout(role_selector_frame)
        role_selector_layout.setContentsMargins(0, 0, 0, 0)
        
        role_templates = [
            "Select a template...",
            "Primary Responder",
            "Critical Analyst",
            "Creative Thinker",
            "Domain Expert",
            "Summarizer",
            "Devil's Advocate",
            "Process Optimizer",
            "User Advocate"
        ]
        
        self.role_template_selector = QComboBox()
        self.role_template_selector.addItems(role_templates)
        self.role_template_selector.setCurrentIndex(0)
        self.role_template_selector.setToolTip("Select a pre-defined role template")
        
        role_selector_layout.addWidget(QLabel("Quick Templates:"))
        role_selector_layout.addWidget(self.role_template_selector)
        
        header_layout.addWidget(role_selector_frame)
        
        # Drag handle and memory toggle
        tools_frame = QFrame()
        tools_layout = QHBoxLayout(tools_frame)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        
        # Memory toggle button
        self.memory_toggle = QToolButton()
        self.memory_toggle.setCheckable(True)
        self.memory_toggle.setText("🧠")
        self.memory_toggle.setChecked(True)
        self.memory_toggle.setToolTip("Toggle agent memory (currently enabled)")
        self.memory_toggle.toggled.connect(self.toggle_memory)
        tools_layout.addWidget(self.memory_toggle)
        
        # Drag handle
        drag_handle = QToolButton()
        drag_handle.setText("≡")
        drag_handle.setToolTip("Drag to reorder agents")
        drag_handle.setCursor(Qt.OpenHandCursor)
        drag_handle.mousePressEvent = self.handle_mouse_press
        tools_layout.addWidget(drag_handle)
        
        header_layout.addWidget(tools_frame)
        layout.addWidget(header_frame)
        
        # Role description entry
        self.role_text_edit = QTextEdit()
        self.role_text_edit.setMaximumHeight(80)
        self.role_text_edit.setPlaceholderText("Describe the agent's role and behavior...")
        self.role_text_edit.setText(self.role_text)
        self.role_text_edit.setToolTip("Enter a description for this agent's role")
        self.role_text_edit.textChanged.connect(self.on_text_changed_internal)
        layout.addWidget(self.role_text_edit)
    
    def toggle_memory(self, enabled):
        """Toggle agent memory on/off"""
        if enabled:
            self.memory_toggle.setToolTip("Toggle agent memory (currently enabled)")
        else:
            self.memory_toggle.setToolTip("Toggle agent memory (currently disabled)")
    
    def on_text_changed_internal(self):
        """Handle text changes in the role description"""
        if self.on_text_changed:
            self.on_text_changed(self.agent_index, self.role_text_edit.toPlainText())
    
    def handle_mouse_press(self, event):
        """Start drag operation when mouse is pressed on drag handle"""
        if event.button() == Qt.LeftButton:
            # Create drag object
            drag = QDrag(self)
            mime_data = QMimeData()
            
            # Store the agent index in the mime data
            mime_data.setData("application/x-agent-index", QByteArray.number(self.agent_index))
            drag.setMimeData(mime_data)
            
            # Start drag operation
            drag.exec_(Qt.MoveAction)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for drops"""
        if event.mimeData().hasFormat("application/x-agent-index"):
            event.acceptProposedAction()
    
    # Fix the dropEvent method in the AgentRoleWidget class
    def dropEvent(self, event):
        # Accept the drop
        event.accept()
        
        # Extract the source index from the mime data
        mime_data = event.mimeData()
        if mime_data.hasFormat("application/x-agent-index"):
            source_index_bytes = mime_data.data("application/x-agent-index")
            # Convert bytes back to integer
            source_index = int(source_index_bytes.data())
            
            # Only move if indices are different
            if source_index != self.agent_index:
                # Call the callback with the source and target indices
                self.on_role_moved(source_index, self.agent_index)
            event.acceptProposedAction()

# ------------------------------------------------------------------------------
# Agent Memory
class AgentMemory:
    """Simple memory store for agents to retain information across turns"""
    
    def __init__(self, max_items=10):
        self.memories = {}  # Agent index -> list of memories
        self.max_items = max_items
    
    def add_memory(self, agent_index, content):
        """Add a new memory item for an agent"""
        if agent_index not in self.memories:
            self.memories[agent_index] = []
        
        # Add memory and trim if needed
        self.memories[agent_index].append(content)
        if len(self.memories[agent_index]) > self.max_items:
            self.memories[agent_index] = self.memories[agent_index][-self.max_items:]
    
    def get_memories(self, agent_index):
        """Get all memories for an agent"""
        return self.memories.get(agent_index, [])
    
    def clear_memory(self, agent_index=None):
        """Clear memory for an agent or all agents"""
        if agent_index is None:
            self.memories = {}
        elif agent_index in self.memories:
            self.memories[agent_index] = []
    
    def summarize_memories(self, agent_index):
        """Summarize memories for an agent"""
        memories = self.get_memories(agent_index)
        if not memories:
            return ""
        
        if len(memories) <= 3:
            return "\n".join(memories)
        else:
            recent = memories[-2:]
            summary = f"From {len(memories)-2} earlier exchanges, you learned: {memories[0]}"
            recent_summary = "\n".join(recent)
            return f"{summary}\n\nMost recent exchanges:\n{recent_summary}"

# ------------------------------------------------------------------------------
# Main Application Window
class GeminiChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenAI Chat Enhanced")
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)
        
        # Set window icon if available
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            logging.warning(f"Could not set window icon: {e}")
        
        # Global variables
        self.num_pages = 20
        self.input_entries = []
        self.output_texts = []
        self.actions_menus = []
        self.button_functions = {}
        self.agent_enabled = False
        self.multi_agent_enabled = False
        self.web_search_enabled = False
        self.active_agents = {}
        self.current_generated_image = None  # Store the generated image
        
        # Add agent memory
        self.agent_memory = AgentMemory()
        
        # Add keyboard shortcuts
        self.setup_shortcuts()
        
        # Set up the main UI first
        self.setup_ui()
        
        # Then apply theme after UI elements exist
        self.apply_theme(current_theme)
        
        # Load unified settings
        load_app_settings(self)
        
        # Initialize mode indicators
        self.update_mode_indicators()
    
    def stop_generation(self):
        """Stop the current text or image generation process"""
        global stop_event
        stop_event.set()
        
        # If there's a generation in progress, try to stop it
        if hasattr(self, 'qt_thread') and self.qt_thread.isRunning():
            self.status_left.setText("Generation stopped by user")
            self.qt_thread.quit()
            self.qt_thread.wait()
        
        # Also stop image generation if it's running
        if hasattr(self, 'img_qt_thread') and self.img_qt_thread.isRunning():
            self.status_left.setText("Image generation stopped by user")
            self.img_qt_thread.quit()
            self.img_qt_thread.wait()
        
        # Stop multi-agent dialog if it's running in continuous mode
        if hasattr(self, 'dialog_worker') and hasattr(self, 'dialog_qt_thread') and self.dialog_qt_thread.isRunning():
            self.status_left.setText("Multi-agent dialog stopped by user")
            # Request graceful stop before forcing thread termination
            if hasattr(self.dialog_worker, 'request_stop'):
                self.dialog_worker.request_stop()
            # Allow a short time for graceful termination
            QTimer.singleShot(2000, lambda: self.force_stop_dialog_worker())
        
        # Update UI
        self.progress_bar.hide()
        Toast.show(self, "Generation stopped", 1500)

    def force_stop_dialog_worker(self):
        """Force stop the dialog worker thread if it's still running"""
        if hasattr(self, 'dialog_qt_thread') and self.dialog_qt_thread.isRunning():
            self.dialog_qt_thread.quit()
            self.dialog_qt_thread.wait()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts for the application"""
        # Generate response shortcut (Ctrl+Return)
        self.generate_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.generate_shortcut.activated.connect(self.generate_current_tab)
        
        # Next tab shortcut (Ctrl+Tab)
        self.next_tab_shortcut = QShortcut(QKeySequence("Ctrl+Tab"), self)
        self.next_tab_shortcut.activated.connect(self.next_tab)
        
        # Previous tab shortcut (Ctrl+Shift+Tab)
        self.prev_tab_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Tab"), self)
        self.prev_tab_shortcut.activated.connect(self.previous_tab)
        
        # Save output shortcut (Ctrl+S)
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(
            lambda: self.perform_file_operation("save")
        )
        
        # Clear history shortcut (Ctrl+L)
        self.clear_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.clear_shortcut.activated.connect(self.clear_current_history)
        
        # Stop generation shortcut (Escape)
        self.stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.stop_shortcut.activated.connect(self.stop_generation)
    
    def generate_current_tab(self):
        """Generate response for the current tab"""
        current_index = self.chat_tabs.currentIndex()
        if current_index >= 0:
            self.generate_response(current_index)
            
    def clear_current_history(self):
        """Clear history for the current tab"""
        current_index = self.chat_tabs.currentIndex()
        if current_index >= 0:
            self.clear_history(current_index)

    def next_tab(self):
        """Switch to the next tab"""
        current = self.chat_tabs.currentIndex()
        next_idx = (current + 1) % self.chat_tabs.count()
        self.chat_tabs.setCurrentIndex(next_idx)
    
    def previous_tab(self):
        """Switch to the previous tab"""
        current = self.chat_tabs.currentIndex()
        prev_idx = (current - 1) % self.chat_tabs.count()
        self.chat_tabs.setCurrentIndex(prev_idx)

    def setup_ui(self):
        """Initialize the main UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        header_label = QLabel("GEMINI Chat Enhanced")
        header_label.setFont(current_fonts["heading"])
        header_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(header_label)
        main_layout.addWidget(header_frame)
        
        # Main splitter (left/right panels)
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (chat tabs)
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        
        # Chat notebook
        self.chat_tabs = QTabWidget()
        self.chat_tabs.setTabsClosable(False)
        self.chat_tabs.setMovable(True)
        self.setup_chat_tabs()
        left_layout.addWidget(self.chat_tabs)
        
        # Right panel (settings)
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        
        # Settings tabs
        self.settings_tabs = QTabWidget()
        self.setup_settings_tabs()
        right_layout.addWidget(self.settings_tabs)
        
        # Add panels to splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([800, 400])  # Initial sizes
        main_layout.addWidget(self.main_splitter)
        
        # Status bar
        self.status_bar = QFrame()
        status_layout = QHBoxLayout(self.status_bar)
        
        self.status_left = QLabel("Ready")
        self.status_left.setFont(QFont("Segoe UI", 9))
        status_layout.addWidget(self.status_left)
        
        status_layout.addStretch()
        
        self.agent_mode_indicator = QLabel("Standard Mode")
        self.agent_mode_indicator.setFont(QFont("Segoe UI", 9))
        status_layout.addWidget(self.agent_mode_indicator)
        
        self.page_indicator = QLabel("Page: 1")
        self.page_indicator.setFont(QFont("Segoe UI", 9))
        status_layout.addWidget(self.page_indicator)
        
        self.model_indicator = QLabel(f"Model: {model_settings['model']}")
        self.model_indicator.setFont(QFont("Segoe UI", 9))
        status_layout.addWidget(self.model_indicator)
        
        main_layout.addWidget(self.status_bar)
        
        # Progress bar
        self.progress_bar = QFrame()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(f"""
            background-color: {themes[current_theme]["bg"]};
            border-radius: 4px;
        """)
        self.progress_indicator = QFrame(self.progress_bar)
        self.progress_indicator.setFixedHeight(8)
        self.progress_indicator.setStyleSheet(f"""
            background-color: {themes[current_theme].get("accent", "#007AFF")};
            border-radius: 4px;
            width: 0%;
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Connect signals
        self.chat_tabs.currentChanged.connect(self.update_page_indicator)
    
    def setup_chat_tabs(self):
        """Create chat tabs with input/output areas"""
        for i in range(self.num_pages):
            # Create page widget
            page = QWidget()
            page_layout = QVBoxLayout(page)
            
            # Input section
            input_frame = QFrame()
            input_layout = QVBoxLayout(input_frame)
            input_label = QLabel("Input:")
            input_label.setFont(current_fonts["label"])
            
            input_edit = QTextEdit()
            input_edit.setFont(current_fonts["input"])
            input_edit.setPlaceholderText("Enter your prompt here...")
            input_edit.setMinimumHeight(100)
            input_edit.setToolTip("Type your message here (Ctrl+Return to send)")
            input_edit.setAccessibleName("Input text area")
            
            input_layout.addWidget(input_label)
            input_layout.addWidget(input_edit)
            page_layout.addWidget(input_frame)
            
            # Actions row
            actions_frame = QFrame()
            actions_layout = QHBoxLayout(actions_frame)
            
            actions_menu = QComboBox()
            actions_menu.addItem("Actions")
            actions_menu.setMinimumWidth(100)
            actions_menu.setToolTip("Select an action to execute")
            actions_menu.setAccessibleName("Actions dropdown")
            # Connect action dropdown to handler
            actions_menu.activated.connect(lambda idx, menu=actions_menu, page_idx=i: self.execute_action(menu, page_idx))
            
            generate_button = QPushButton("Generate")
            generate_button.setMinimumWidth(100)
            generate_button.setToolTip("Generate a response (Ctrl+Return)")
            generate_button.setAccessibleName("Generate response button")
            generate_button.clicked.connect(lambda checked, idx=i: self.generate_response(idx))
            
            stop_button = QPushButton("Stop")
            stop_button.setMinimumWidth(100)
            stop_button.setToolTip("Stop generation (Esc)")
            stop_button.setAccessibleName("Stop generation button")
            stop_button.clicked.connect(self.stop_generation)
            
            manage_actions_button = QPushButton("Manage Actions")
            manage_actions_button.setMinimumWidth(100)
            manage_actions_button.setToolTip("Create or edit custom actions")
            manage_actions_button.setAccessibleName("Manage actions button")
            manage_actions_button.clicked.connect(self.open_button_manager)
            
            clear_history_button = QPushButton("Clear History")
            clear_history_button.setMinimumWidth(100)
            clear_history_button.setToolTip("Clear conversation history (Ctrl+L)")
            clear_history_button.setAccessibleName("Clear history button")
            clear_history_button.clicked.connect(lambda checked, idx=i: self.clear_history(idx))
            
            # Mode indicator label
            mode_label = QLabel("")
            
            actions_layout.addWidget(actions_menu)
            actions_layout.addWidget(generate_button)
            actions_layout.addWidget(stop_button)
            actions_layout.addWidget(manage_actions_button)
            actions_layout.addWidget(clear_history_button)
            actions_layout.addStretch()
            actions_layout.addWidget(mode_label)
            
            page_layout.addWidget(actions_frame)
            
            # Output section
            output_frame = QFrame()
            output_layout = QVBoxLayout(output_frame)
            output_label = QLabel("Output:")
            output_label.setFont(current_fonts["label"])
            
            output_text = QTextEdit()
            output_text.setFont(current_fonts["output"])
            output_text.setReadOnly(True)
            output_text.setToolTip("Response will appear here")
            output_text.setAccessibleName("Output text area")
            
            output_layout.addWidget(output_label)
            output_layout.addWidget(output_text)
            
            page_layout.addWidget(output_frame, 1)  # Give output section more space
            
            # Add to collections
            self.input_entries.append(input_edit)
            self.output_texts.append(output_text)
            self.actions_menus.append(actions_menu)
            
            # Add the page to tabs
            self.chat_tabs.addTab(page, f"Page {i+1}")
    
    def setup_settings_tabs(self):
        """Create the settings tabs on the right panel"""
        # Chat Settings Tab
        chat_settings_tab = QWidget()
        chat_settings_layout = QVBoxLayout(chat_settings_tab)
        
        # Multi-response mode
        multi_response_frame = QFrame()
        multi_response_layout = QHBoxLayout(multi_response_frame)
        
        self.multi_response_check = QCheckBox("Multi Response Mode")
        self.multi_response_check.toggled.connect(self.toggle_multi_response_mode)
        
        self.response_count_label = QLabel("Number of responses:")
        self.response_count_label.setVisible(False)
        
        self.response_count_combo = QComboBox()
        for i in range(1, 11):
            self.response_count_combo.addItem(str(i))
        self.response_count_combo.setCurrentIndex(0)
        self.response_count_combo.setVisible(False)
        
        multi_response_layout.addWidget(self.multi_response_check)
        multi_response_layout.addWidget(self.response_count_label)
        multi_response_layout.addWidget(self.response_count_combo)
        multi_response_layout.addStretch()
        
        chat_settings_layout.addWidget(multi_response_frame)
        
        # Web Search Mode
        web_search_frame = QFrame()
        web_search_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        web_search_layout = QVBoxLayout(web_search_frame)
        
        web_search_layout.addWidget(QLabel("Web Search"))
        
        self.web_search_check = QCheckBox("Enable Web Search")
        self.web_search_check.toggled.connect(self.toggle_web_search_mode)
        
        web_search_info = QLabel("When enabled, models with search capability\nwill search the web for up-to-date information.")
        web_search_info.setWordWrap(True)
        
        search_status_frame = QFrame()
        search_status_layout = QHBoxLayout(search_status_frame)
        
        self.search_status_indicator = QLabel("●")
        self.search_status_indicator.setStyleSheet("color: #8E8E93;")
        
        self.search_status_label = QLabel("Web search inactive")
        
        search_status_layout.addWidget(self.search_status_indicator)
        search_status_layout.addWidget(self.search_status_label)
        search_status_layout.addStretch()
        
        web_search_layout.addWidget(self.web_search_check)
        web_search_layout.addWidget(web_search_info)
        web_search_layout.addWidget(search_status_frame)
        
        chat_settings_layout.addWidget(web_search_frame)
        
        # Model Settings
        settings_frame = QFrame()
        settings_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        settings_layout = QVBoxLayout(settings_frame)
        
        settings_layout.addWidget(QLabel("Model Settings"))
        
        # Temperature
        temp_frame = QFrame()
        temp_layout = QHBoxLayout(temp_frame)
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temperature_entry = QLineEdit(str(model_settings.get("temperature", 0.7)))
        temp_layout.addWidget(self.temperature_entry)
        settings_layout.addWidget(temp_frame)

        # Top P
        top_p_frame = QFrame()
        top_p_layout = QHBoxLayout(top_p_frame)
        top_p_layout.addWidget(QLabel("Top P:"))
        self.top_p_entry = QLineEdit(str(model_settings.get("top_p", 1.0)))
        top_p_layout.addWidget(self.top_p_entry)
        settings_layout.addWidget(top_p_frame)

        # Max Tokens (replacing Top K for OpenAI)
        max_tokens_frame = QFrame()
        max_tokens_layout = QHBoxLayout(max_tokens_frame)
        max_tokens_layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens_entry = QLineEdit(str(model_settings.get("max_tokens", 4096)))
        self.max_tokens_entry.setToolTip("Maximum tokens in completion (higher values allow longer responses)")
        max_tokens_layout.addWidget(self.max_tokens_entry)
        settings_layout.addWidget(max_tokens_frame)

        # Add thinking process toggle
        thinking_frame = QFrame()
        thinking_layout = QHBoxLayout(thinking_frame)
        self.show_thinking_checkbox = QCheckBox("Show Step-by-Step Thinking")
        self.show_thinking_checkbox.setToolTip("Enables explicit reasoning steps")
        self.show_thinking_checkbox.setChecked(model_settings.get("show_thinking", False))
        thinking_layout.addWidget(self.show_thinking_checkbox)
        settings_layout.addWidget(thinking_frame)
        
        # Model Selection - Updated for OpenAI models
        model_frame = QFrame()
        model_layout = QHBoxLayout(model_frame)
        model_layout.addWidget(QLabel("Select Model:"))
        
        self.model_selector = QComboBox()
        models = [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "gpt-4o-mini-search-preview",
            "o3-mini",
            "o1-mini",
        ]
        self.model_selector.addItems(models)
        self.model_selector.setCurrentText(model_settings["model"])
        self.model_selector.currentIndexChanged.connect(self.update_parameter_states)
        
        model_layout.addWidget(self.model_selector)
        settings_layout.addWidget(model_frame)
        
        # Apply button
        self.apply_settings_button = QPushButton("Apply Settings")
        self.apply_settings_button.clicked.connect(self.apply_settings)
        settings_layout.addWidget(self.apply_settings_button)
        
        chat_settings_layout.addWidget(settings_frame)
        chat_settings_layout.addStretch()
        
        # Add to settings tab widget
        self.settings_tabs.addTab(chat_settings_tab, "Chat")
        
        # Instructions Tab
        instructions_tab = QWidget()
        instructions_layout = QVBoxLayout(instructions_tab)
        
        # System Instructions
        system_frame = QFrame()
        system_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        system_layout = QVBoxLayout(system_frame)
        
        system_layout.addWidget(QLabel("System Instructions"))
        system_layout.addWidget(QLabel("Set default system instructions:"))
        
        self.system_instructions_text = QTextEdit()
        self.system_instructions_text.setFont(current_fonts["input"])
        self.system_instructions_text.setMinimumHeight(150)
        system_layout.addWidget(self.system_instructions_text)
        
        instructions_layout.addWidget(system_frame)
        
        # Developer Instructions
        developer_frame = QFrame()
        developer_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        developer_layout = QVBoxLayout(developer_frame)
        
        developer_layout.addWidget(QLabel("Developer Instructions"))
        developer_layout.addWidget(QLabel("Set developer instructions:"))
        
        self.developer_instructions_text = QTextEdit()
        self.developer_instructions_text.setFont(current_fonts["input"])
        self.developer_instructions_text.setMinimumHeight(150)
        developer_layout.addWidget(self.developer_instructions_text)
        
        instructions_layout.addWidget(developer_frame)
        
        # Save button
        self.save_instructions_button = QPushButton("Save Instructions")
        self.save_instructions_button.clicked.connect(self.save_system_instructions)
        instructions_layout.addWidget(self.save_instructions_button)
        
        # Instruction presets
        preset_frame = QFrame()
        preset_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        preset_layout = QHBoxLayout(preset_frame)
        
        preset_layout.addWidget(QLabel("Instruction Presets:"))
        
        self.preset_dropdown = QComboBox()
        presets = [
            "General Assistant",
            "Code Assistant",
            "Creative Writer"
        ]
        self.preset_dropdown.addItems(presets)
        
        preset_layout.addWidget(self.preset_dropdown)
        
        self.apply_preset_btn = QPushButton("Apply")
        self.apply_preset_btn.clicked.connect(self.apply_preset)
        preset_layout.addWidget(self.apply_preset_btn)
        
        instructions_layout.addWidget(preset_frame)
        instructions_layout.addStretch()
        
        self.settings_tabs.addTab(instructions_tab, "Instructions")
        
        # Add other tabs (Appearance, Tools, Agents)
        self.add_appearance_tab()
        self.add_tools_tab()
        self.add_agents_tab()
    
    def add_appearance_tab(self):
        """Add appearance customization tab"""
        appearance_tab = QWidget()
        appearance_layout = QVBoxLayout(appearance_tab)
        
        # Theme selector
        theme_frame = QFrame()
        theme_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        theme_layout = QVBoxLayout(theme_frame)
        
        theme_layout.addWidget(QLabel("Theme"))
        
        self.theme_buttons = {}
        theme_group = QFrame()
        theme_group_layout = QVBoxLayout(theme_group)
        
        for theme_name in themes.keys():
            theme_radio = QCheckBox(theme_name)
            if theme_name == current_theme:
                theme_radio.setChecked(True)
            theme_radio.toggled.connect(lambda checked, tn=theme_name: self.apply_theme(tn) if checked else None)
            self.theme_buttons[theme_name] = theme_radio
            theme_group_layout.addWidget(theme_radio)
        
        theme_layout.addWidget(theme_group)
        appearance_layout.addWidget(theme_frame)
        
        # Quick Font Settings
        font_frame = QFrame()
        font_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        font_layout = QVBoxLayout(font_frame)
        
        font_layout.addWidget(QLabel("Quick Font Settings"))
        
        # Text size
        size_frame = QFrame()
        size_layout = QHBoxLayout(size_frame)
        
        size_layout.addWidget(QLabel("Text Size:"))
        
        increase_size_btn = QPushButton("Larger")
        increase_size_btn.clicked.connect(lambda: self.change_font_size(1))
        
        decrease_size_btn = QPushButton("Smaller")
        decrease_size_btn.clicked.connect(lambda: self.change_font_size(-1))
        
        size_layout.addWidget(increase_size_btn)
        size_layout.addWidget(decrease_size_btn)
        size_layout.addStretch()
        
        font_layout.addWidget(size_frame)
        
        # Font family
        family_frame = QFrame()
        family_layout = QHBoxLayout(family_frame)
        
        family_layout.addWidget(QLabel("Font Family:"))
        
        self.font_family_dropdown = QComboBox()
        font_families = ["Segoe UI", "Arial", "Helvetica", "Tahoma", "Verdana", "SF Pro Text"]
        self.font_family_dropdown.addItems(font_families)
        self.font_family_dropdown.setCurrentText("Segoe UI")
        self.font_family_dropdown.currentTextChanged.connect(self.change_font_family)
        
        family_layout.addWidget(self.font_family_dropdown)
        family_layout.addStretch()
        
        font_layout.addWidget(family_frame)
        appearance_layout.addWidget(font_frame)
        
        # Advanced customization
        advanced_button = QPushButton("Advanced Customization")
        advanced_button.clicked.connect(self.open_gui_customizer)
        appearance_layout.addWidget(advanced_button)
        
        # Reset layout
        reset_layout_button = QPushButton("Reset to Default")
        reset_layout_button.clicked.connect(self.reset_layout)
        appearance_layout.addWidget(reset_layout_button)
        
        appearance_layout.addStretch()
        
        self.settings_tabs.addTab(appearance_tab, "Appearance")
    def update_interaction_controls(self):
        """Update UI controls based on selected interaction mode"""
        mode = self.interaction_mode_selector.currentText()
        continuous_mode = (mode == "Continuous Debate")
        
        # Show/hide turn limit controls based on continuous mode
        self.turn_limit_spinner.setEnabled(continuous_mode)
        # Find and update the turn limit label
        for i in range(self.interaction_mode_selector.parentWidget().layout().count()):
            widget = self.interaction_mode_selector.parentWidget().layout().itemAt(i).widget()
            if isinstance(widget, QLabel) and widget.text() == "Turn Limit:":
                widget.setEnabled(continuous_mode)

    def add_tools_tab(self):
        """Add tools tab with image generation and file operations"""
        tools_tab = QWidget()
        tools_layout = QVBoxLayout(tools_tab)
        
        # Environment section
        env_frame = QFrame()
        env_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        env_layout = QVBoxLayout(env_frame)
        
        env_layout.addWidget(QLabel("Environment"))
        
        load_env_button = QPushButton("Load .env File")
        load_env_button.clicked.connect(self.load_env_file)
        env_layout.addWidget(load_env_button)
        
        tools_layout.addWidget(env_frame)
        
        # Image Generation section - Updated for DALL-E
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        image_layout = QVBoxLayout(image_frame)
        
        image_layout.addWidget(QLabel("Image Generation"))
        image_layout.addWidget(QLabel("Enter image prompt:"))
        
        self.photo_prompt_entry = QLineEdit()
        image_layout.addWidget(self.photo_prompt_entry)
        
        # Add model selection for image generation
        model_frame = QFrame()
        model_layout = QHBoxLayout(model_frame)
        model_layout.addWidget(QLabel("Model:"))
        
        self.image_model_selector = QComboBox()
        self.image_model_selector.addItems([
            "DALL-E 3",
            "DALL-E 2"
        ])
        model_layout.addWidget(self.image_model_selector)
        image_layout.addWidget(model_frame)
        
        # Add image size selection (for Imagen)
        size_frame = QFrame()
        size_layout = QHBoxLayout(size_frame)
        size_layout.addWidget(QLabel("Size:"))
        
        self.image_size_selector = QComboBox()
        self.image_size_selector.addItems([
            "1024x1024", 
            "1024x768",
            "768x1024"
        ])
        size_layout.addWidget(self.image_size_selector)
        image_layout.addWidget(size_frame)
        
        generate_photo_button = QPushButton("Generate Image")
        generate_photo_button.clicked.connect(self.generate_photo)
        image_layout.addWidget(generate_photo_button)
        
        # Add image display area
        self.image_display = QLabel("Image will appear here")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumHeight(256)
        self.image_display.setStyleSheet("background-color: #2c2c2c; border-radius: 5px;")
        image_layout.addWidget(self.image_display)
        
        # Add save image button
        save_image_button = QPushButton("Save Image")
        save_image_button.clicked.connect(self.save_generated_image)
        save_image_button.setEnabled(False)
        self.save_image_button = save_image_button
        image_layout.addWidget(save_image_button)
        
        tools_layout.addWidget(image_frame)
        
        # File Operations section
        file_frame = QFrame()
        file_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        file_layout = QVBoxLayout(file_frame)
        
        file_layout.addWidget(QLabel("File Operations"))
        
        load_file_button = QPushButton("Load Text from File")
        load_file_button.clicked.connect(lambda: self.perform_file_operation("load"))
        file_layout.addWidget(load_file_button)
        
        save_file_button = QPushButton("Save Output to File")
        save_file_button.clicked.connect(lambda: self.perform_file_operation("save"))
        file_layout.addWidget(save_file_button)
        
        save_session_button = QPushButton("Save Current Session")
        save_session_button.clicked.connect(lambda: self.perform_file_operation("save_session"))
        file_layout.addWidget(save_session_button)
        
        load_session_button = QPushButton("Load Saved Session")
        load_session_button.clicked.connect(lambda: self.perform_file_operation("load_session"))
        file_layout.addWidget(load_session_button)
        
        tools_layout.addWidget(file_frame)
        
        # AI Goals section
        goals_frame = QFrame()
        goals_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        goals_layout = QVBoxLayout(goals_frame)
        
        goals_layout.addWidget(QLabel("AI Goals"))
        
        update_goals_button = QPushButton("Update AI Goals")
        update_goals_button.clicked.connect(self.update_goals)
        goals_layout.addWidget(update_goals_button)
        
        view_goals_button = QPushButton("View AI Goals")
        view_goals_button.clicked.connect(self.view_goals)
        goals_layout.addWidget(view_goals_button)
        
        tools_layout.addWidget(goals_frame)
        tools_layout.addStretch()
        
        self.settings_tabs.addTab(tools_tab, "Tools")
    
    

    def add_agents_tab(self):
        """Add agents configuration tab with enhanced features"""
        agents_tab = QScrollArea()
        agents_tab.setWidgetResizable(True)
        agents_widget = QWidget()
        agents_layout = QVBoxLayout(agents_widget)
        
        # Agent Mode section
        agent_frame = QFrame()
        agent_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        agent_layout = QVBoxLayout(agent_frame)
        
        agent_layout.addWidget(QLabel("Agent Mode"))
        
        # Toggle and status
        toggle_frame = QFrame()
        toggle_layout = QHBoxLayout(toggle_frame)
        
        self.agent_toggle = QCheckBox("Enable Agent Mode")
        self.agent_toggle.toggled.connect(self.toggle_agent_mode)
        self.agent_toggle.setToolTip("Use an AI agent with specialized behaviors")
        toggle_layout.addWidget(self.agent_toggle)
        
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        
        self.agent_status_indicator = QLabel("●")
        self.agent_status_indicator.setStyleSheet("color: #8E8E93;")
        
        self.agent_status_label = QLabel("Agent mode inactive")
        
        status_layout.addWidget(self.agent_status_indicator)
        status_layout.addWidget(self.agent_status_label)
        
        toggle_layout.addWidget(status_frame)
        toggle_layout.addStretch()
        
        agent_layout.addWidget(toggle_frame)
        
        # Settings
        settings_frame = QFrame()
        settings_layout = QHBoxLayout(settings_frame)
        
        left_settings = QFrame()
        left_layout = QVBoxLayout(left_settings)
        
        model_frame = QFrame()
        model_layout = QHBoxLayout(model_frame)
        model_layout.addWidget(QLabel("Agent Model:"))
        
        self.agent_model_selector = QComboBox()
        self.agent_model_selector.addItems([
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4o-mini",
        ])
        self.agent_model_selector.setToolTip("Select which OpenAI model to use for agent processing")
        model_layout.addWidget(self.agent_model_selector)
        
        left_layout.addWidget(model_frame)
        
        right_settings = QFrame()
        right_layout = QVBoxLayout(right_settings)
        
        name_frame = QFrame()
        name_layout = QHBoxLayout(name_frame)
        name_layout.addWidget(QLabel("Agent Name:"))
        
        self.agent_name_entry = QLineEdit("Research Assistant")
        self.agent_name_entry.setToolTip("Name for your agent - will be used in its instructions")
        name_layout.addWidget(self.agent_name_entry)
        
        right_layout.addWidget(name_frame)
        
        settings_layout.addWidget(left_settings)
        settings_layout.addWidget(right_settings)
        
        agent_layout.addWidget(settings_frame)
        agents_layout.addWidget(agent_frame)
        
        # Agent Instructions
        agent_desc_frame = QFrame()
        agent_desc_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        agent_desc_layout = QVBoxLayout(agent_desc_frame)
        
        agent_desc_layout.addWidget(QLabel("Agent Instructions"))
        
        self.agent_description = QTextEdit()
        self.agent_description.setMinimumHeight(100)
        self.agent_description.setText("You are a research assistant that helps users find information and answers questions using the most up-to-date information available.")
        self.agent_description.setToolTip("Instructions for how your agent should behave")
        agent_desc_layout.addWidget(self.agent_description)
        
        # Presets
        preset_frame = QFrame()
        preset_layout = QHBoxLayout(preset_frame)
        
        preset_layout.addWidget(QLabel("Quick Templates:"))
        
        self.agent_preset_dropdown = QComboBox()
        agent_presets = [
            "Research Assistant",
            "Code Assistant",
            "Travel Planner",
            "Product Researcher"
        ]
        self.agent_preset_dropdown.addItems(agent_presets)
        self.agent_preset_dropdown.setToolTip("Choose a pre-defined agent role")
        preset_layout.addWidget(self.agent_preset_dropdown)
        
        apply_preset_button = QPushButton("Apply")
        apply_preset_button.setToolTip("Apply the selected template")
        apply_preset_button.clicked.connect(self.apply_agent_preset)
        preset_layout.addWidget(apply_preset_button)
        
        agent_desc_layout.addWidget(preset_frame)
        
        agent_explanation = QLabel("The agent uses OpenAI's API for reasoning and tool use capabilities.")
        agent_explanation.setWordWrap(True)
        agent_desc_layout.addWidget(agent_explanation)
        
        agents_layout.addWidget(agent_desc_frame)
        
        # Agent Tools - Enhanced with more tools
        tools_frame = QFrame()
        tools_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        tools_layout = QVBoxLayout(tools_frame)
        
        tools_layout.addWidget(QLabel("Agent Capabilities"))
        
        # Basic capabilities
        self.web_browsing_check = QCheckBox("Web Information (when available)")
        self.web_browsing_check.setChecked(True)
        self.web_browsing_check.setToolTip("Allow agent to access up-to-date web information")
        tools_layout.addWidget(self.web_browsing_check)
        
        self.code_interpreter_check = QCheckBox("Code Generation")
        self.code_interpreter_check.setChecked(True)
        self.code_interpreter_check.setToolTip("Allow agent to write and analyze code")
        tools_layout.addWidget(self.code_interpreter_check)
        
        self.reasoning_check = QCheckBox("Step-by-step Reasoning")
        self.reasoning_check.setChecked(True)
        self.reasoning_check.setToolTip("Have agent show its reasoning process")
        tools_layout.addWidget(self.reasoning_check)
        
        # Additional tools
        self.calculator_check = QCheckBox("Calculator Tool")
        self.calculator_check.setChecked(True)
        self.calculator_check.setToolTip("Allow agent to perform calculations")
        tools_layout.addWidget(self.calculator_check)
        
        self.calendar_check = QCheckBox("Calendar and Date Tool")
        self.calendar_check.setChecked(True)
        self.calendar_check.setToolTip("Allow agent to use date/time information")
        tools_layout.addWidget(self.calendar_check)
        
        self.search_check = QCheckBox("Specialized Search Tools")
        self.search_check.setToolTip("Allow agent to use specialized search capabilities")
        tools_layout.addWidget(self.search_check)
        
        agents_layout.addWidget(tools_frame)
        
        # Multi-Agent Dialog - Enhanced with drag-and-drop
        multi_agent_frame = QFrame()
        multi_agent_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        multi_agent_layout = QVBoxLayout(multi_agent_frame)
        
        multi_agent_layout.addWidget(QLabel("Multi-Agent Dialog"))
        
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        
        self.multi_agent_toggle = QCheckBox("Enable Multi-Agent Dialog")
        self.multi_agent_toggle.toggled.connect(self.toggle_multi_agent_mode)
        self.multi_agent_toggle.setToolTip("Enable conversations between multiple specialized AI agents")
        header_layout.addWidget(self.multi_agent_toggle)
        
        count_frame = QFrame()
        count_layout = QHBoxLayout(count_frame)
        count_layout.addWidget(QLabel("Agents:"))
        
        self.agent_count_spinner = QSpinBox()
        self.agent_count_spinner.setRange(2, 6)
        self.agent_count_spinner.setValue(3)
        self.agent_count_spinner.setEnabled(False)
        self.agent_count_spinner.setToolTip("Number of agents in the conversation")
        self.agent_count_spinner.valueChanged.connect(self.update_agent_roles_ui)
        count_layout.addWidget(self.agent_count_spinner)
        
        header_layout.addWidget(count_frame)
        
        # Add interaction mode selector
        interaction_frame = QFrame()
        interaction_layout = QHBoxLayout(interaction_frame)
        interaction_layout.addWidget(QLabel("Interaction:"))
        
        self.interaction_mode_selector = QComboBox()
        self.interaction_mode_selector.addItems([
            "Sequential",
            "Interactive",
            "Continuous Debate"  # Add this option
        ])
        self.interaction_mode_selector.setToolTip("How agents interact with each other")
        self.interaction_mode_selector.setEnabled(False)
        interaction_layout.addWidget(self.interaction_mode_selector)
        
        # Add turn limit for continuous mode
        self.turn_limit_spinner = QSpinBox()
        self.turn_limit_spinner.setRange(0, 100)  # 0 means unlimited
        self.turn_limit_spinner.setValue(0)
        self.turn_limit_spinner.setEnabled(False)
        self.turn_limit_spinner.setToolTip("Maximum conversation turns (0 = unlimited)")
        
        turn_limit_label = QLabel("Turn Limit:")
        turn_limit_label.setEnabled(False)
        
        interaction_layout.addWidget(turn_limit_label)
        interaction_layout.addWidget(self.turn_limit_spinner)
        
        # Connect interaction mode changes to UI updates
        self.interaction_mode_selector.currentIndexChanged.connect(self.update_interaction_controls)
        
        header_layout.addWidget(interaction_frame)
        header_layout.addStretch()
        
        multi_agent_layout.addWidget(header_frame)
        
        # Agent roles container
        info_label = QLabel("Multi-agent mode enables a conversation between specialized AI personas. Drag and drop to reorder agents.")
        info_label.setWordWrap(True)
        multi_agent_layout.addWidget(info_label)
        
        roles_header_frame = QFrame()
        roles_header_layout = QHBoxLayout(roles_header_frame)
        
        roles_header = QLabel("Agent Roles")
        roles_header.setFont(current_fonts["label"])
        roles_header_layout.addWidget(roles_header)
        
        # Add buttons for role management
        clear_memories_btn = QPushButton("Clear All Memories")
        clear_memories_btn.setToolTip("Clear stored memories for all agents")
        clear_memories_btn.clicked.connect(self.clear_all_agent_memories)
        roles_header_layout.addWidget(clear_memories_btn)
        
        reset_roles_btn = QPushButton("Reset Roles")
        reset_roles_btn.setToolTip("Reset agent roles to defaults")
        reset_roles_btn.clicked.connect(self.reset_agent_roles)
        roles_header_layout.addWidget(reset_roles_btn)
        
        multi_agent_layout.addWidget(roles_header_frame)
        
        self.agent_roles_container = QFrame()
        self.agent_roles_container.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.agent_roles_layout = QVBoxLayout(self.agent_roles_container)
        multi_agent_layout.addWidget(self.agent_roles_container)
        
        # Initialize roles
        self.agent_roles = {}
        self.agent_role_entries = {}
        self.agent_role_widgets = {}
        self.default_agent_roles = [
            "Primary agent responding directly to the user's query with detailed information.",
            "Critical analyst reviewing the first agent's response, adding additional context or corrections.",
            "Synthesizer combining insights from previous agents and suggesting next steps or actionable items.",
            "Expert consultant providing specialized domain knowledge on the topic at hand.",
            "Summarizer distilling the key points from all agents into a concise final response.",
            "Creative thinker exploring unusual angles and innovative solutions to the problem."
        ]
        
        # Initialize agent roles UI
        self.update_agent_roles_ui()
        
        agents_layout.addWidget(multi_agent_frame)
        agents_layout.addStretch()
        
        agents_tab.setWidget(agents_widget)
        self.settings_tabs.addTab(agents_tab, "Agents")
    
    def update_agent_roles_ui(self):
        """Update the UI to show the correct number of agent role configuration fields with drag-drop support"""
        # Clear existing widgets
        for widget in self.agent_role_widgets.values():
            widget.setParent(None)
        
        self.agent_role_entries.clear()
        self.agent_role_widgets.clear()
        
        # Get the current number of agents
        num_agents = self.agent_count_spinner.value()
        
        # Create a role entry for each agent using the enhanced draggable widgets
        for i in range(num_agents):
            # Get the current role text for this agent
            if i < len(self.default_agent_roles):
                role_text = self.agent_roles.get(i, self.default_agent_roles[i])
            else:
                role_text = self.agent_roles.get(i, f"Agent {i+1} analyzing and responding to previous content.")
            
            # Create the widget with callbacks
            role_widget = AgentRoleWidget(
                self.agent_roles_container,
                i,
                role_text,
                on_text_changed=self.update_agent_role_text,
                on_role_moved=self.move_agent_role
            )
            
            # Enable/disable based on multi-agent mode
            role_widget.setEnabled(self.multi_agent_enabled)
            
            # Add to container
            self.agent_roles_layout.addWidget(role_widget)
            
            # Store references
            self.agent_role_widgets[i] = role_widget
            self.agent_role_entries[i] = role_widget.role_text_edit
            self.agent_roles[i] = role_text
        
        # Update the interaction mode dropdown state
        self.interaction_mode_selector.setEnabled(self.multi_agent_enabled)
        
        # After creating all the role entries, update mode indicators if in multi-agent mode
        if self.agent_enabled and self.multi_agent_enabled:
            self.update_mode_indicators()
    
    def update_agent_role_text(self, agent_index, text):
        """Update the stored text for an agent role"""
        self.agent_roles[agent_index] = text
    
    def move_agent_role(self, source_index, target_index):
        """Handle moving agent roles via drag and drop"""
        # Ensure both indices are integers
        source_index = int(source_index) if not isinstance(source_index, int) else source_index
        target_index = int(target_index) if not isinstance(target_index, int) else target_index
        
        # Store the roles temporarily
        roles_copy = self.agent_roles.copy()
        
        # If source is before target, shift everything between down
        if source_index < target_index:
            for i in range(source_index, target_index):
                self.agent_roles[i] = roles_copy.get(i+1, "")
            self.agent_roles[target_index] = roles_copy.get(source_index, "")
        
        # If source is after target, shift everything between up
        elif source_index > target_index:
            for i in range(source_index, target_index, -1):
                self.agent_roles[i] = roles_copy.get(i-1, "")
            self.agent_roles[target_index] = roles_copy.get(source_index, "")
        
        # Rebuild the UI with the new order
        self.update_agent_roles_ui()
        
        # Show confirmation
        Toast.show(self, f"Moved Agent {source_index + 1} to position {target_index + 1}", 1500)
    
    def clear_all_agent_memories(self):
        """Clear all stored memories for agents"""
        self.agent_memory.clear_memory()
        Toast.show(self, "All agent memories cleared", 1500)
    
    def reset_agent_roles(self):
        """Reset agent roles to defaults"""
        num_agents = self.agent_count_spinner.value()
        self.agent_roles = {}
        
        for i in range(min(num_agents, len(self.default_agent_roles))):
            self.agent_roles[i] = self.default_agent_roles[i]
        
        self.update_agent_roles_ui()
        Toast.show(self, "Agent roles reset to defaults", 1500)

    # Helper Functions
    def apply_theme(self, theme_name):
        """Apply selected theme to the application"""
        global current_theme
        if (theme_name in themes):
            current_theme = theme_name
            
            # Update checkboxes
            if hasattr(self, 'theme_buttons'):
                for name, btn in self.theme_buttons.items():
                    btn.setChecked(name == theme_name)
            
            # Apply theme colors to app
            theme = themes[theme_name]
            
            # Main window background
            # Update text inputs
            for input_entry in self.input_entries:
                input_entry.setStyleSheet(f"""
                    background-color: {theme['input_bg']};
                    color: {theme['fg']};
                    border: 1px solid {theme['border']};
                    border-radius: 5px;
                    padding: 8px;
                """)
            
            # Update text outputs
            for output_text in self.output_texts:
                output_text.setStyleSheet(f"""
                    background-color: {theme['output_bg']};
                    color: {theme['fg']};
                    border: 1px solid {theme['border']};
                    border-radius: 5px;
                    padding: 8px;
                """)
            
            # Update system and developer instruction fields if they exist
            if hasattr(self, 'system_instructions_text') and hasattr(self, 'developer_instructions_text'):
                for widget in [self.system_instructions_text, self.developer_instructions_text]:
                    widget.setStyleSheet(f"""
                        background-color: {theme['input_bg']};
                        color: {theme['fg']};
                        border: 1px solid {theme['border']};
                        border-radius: 5px;
                        padding: 8px;
                    """)
            
            # Update agent description if it exists
            if hasattr(self, 'agent_description'):
                self.agent_description.setStyleSheet(f"""
                    background-color: {theme['input_bg']};
                    color: {theme['fg']};
                    border: 1px solid {theme['border']};
                    border-radius: 5px;
                    padding: 8px;
                """)
            
            # Update progress bar
            self.progress_indicator.setStyleSheet(f"""
                background-color: {theme.get('accent', '#007AFF')};
                border-radius: 4px;
            """)
    
    def update_all_fonts(self):
        """Update fonts throughout the application"""
        # Update input fields
        for input_entry in self.input_entries:
            input_entry.setFont(current_fonts["input"])
        
        # Update output fields
        for output_text in self.output_texts:
            output_text.setFont(current_fonts["output"])
        
        # Update system and developer instruction fields if they exist
        if hasattr(self, 'system_instructions_text'):
            self.system_instructions_text.setFont(current_fonts["input"])
        if hasattr(self, 'developer_instructions_text'):
            self.developer_instructions_text.setFont(current_fonts["input"])
        
        # Update agent description if it exists
        if hasattr(self, 'agent_description'):
            self.agent_description.setFont(current_fonts["input"])
    
    def change_font_size(self, delta):
        """Change all font sizes by delta amount"""
        global current_fonts
        
        for key, font in current_fonts.items():
            current_size = font.pointSize()
            new_size = max(8, current_size + delta)  # Don't go below size 8
            new_font = QFont(font.family(), new_size)
            if font.bold():
                new_font.setBold(True)
            if font.italic():
                new_font.setItalic(True)
            current_fonts[key] = new_font
        
        self.update_all_fonts()
        self.save_user_preferences()
        
        # Show toast
        Toast.show(self, "Font size updated", 1500)
    
    def change_font_family(self, family):
        """Change all font families"""
        global current_fonts
        
        for key, font in current_fonts.items():
            new_font = QFont(family, font.pointSize())
            if font.bold():
                new_font.setBold(True)
            if font.italic():
                new_font.setItalic(True)
            current_fonts[key] = new_font
        
        self.update_all_fonts()
        self.save_user_preferences()
        
        # Show toast
        Toast.show(self, f"Font changed to {family}", 1500)
    
    def toggle_multi_response_mode(self, enabled):
        """Handle toggling multi-response mode on and off"""
        self.response_count_label.setVisible(enabled)
        self.response_count_combo.setVisible(enabled)
    
    def toggle_web_search_mode(self, enabled):
        """Handle toggling web search mode on and off"""
        self.web_search_enabled = enabled
        self.update_search_status_indicator()
        
        if enabled:
            # When enabled, show a message and suggest switching to a search-capable model
            current_model = self.model_selector.currentText()
            if not current_model.endswith("search-preview"):
                QMessageBox.information(
                    self,
                    "Web Search Mode Enabled",
                    "Web search mode is now enabled. For best results, please select one of the search models:\n\n"
                    "• gpt-4o-search-preview\n"
                    "• gpt-4o-mini-search-preview"
                )
        else:
            # When disabled, inform the user
            QMessageBox.information(
                self,
                "Web Search Mode Disabled",
                "Web search has been disabled. The model will no longer search the web for information."
            )
    
    def update_search_status_indicator(self):
        """Update the visual indicator for search status"""
        if self.web_search_enabled:
            if self.model_selector.currentText().endswith("search-preview"):
                # Green dot for active and compatible
                self.search_status_indicator.setStyleSheet("color: #34C759;")
                self.search_status_label.setText("Web search active")
            else:
                # Yellow dot for active but model might not be compatible
                self.search_status_indicator.setStyleSheet("color: #FF9500;")
                self.search_status_label.setText("Web search enabled (use search model)")
        else:
            # Gray dot for inactive
            self.search_status_indicator.setStyleSheet("color: #8E8E93;")
            self.search_status_label.setText("Web search inactive")
    
    def update_parameter_states(self):
        """Update parameter field states based on selected model"""
        current_model = self.model_selector.currentText()
        is_search_model = current_model.endswith("search-preview")
        is_thinking_capable = True  # All OpenAI models can do thinking
        is_o_family = is_o_family_model(current_model)
        
        # Visual indication for incompatible parameters with search models
        self.top_p_entry.setEnabled(not is_search_model and not is_o_family)
        
        # Update UI labels for model-specific features
        if is_o_family:
            self.top_p_entry.setToolTip("top_p parameter not used with O-family models")
            self.developer_instructions_text.setToolTip("Developer instructions apply to O-family models")
        else:
            self.top_p_entry.setToolTip("Controls diversity via nucleus sampling (0.0 to 1.0)")
            self.developer_instructions_text.setToolTip("Developer instructions only apply to O-family models")
        
        # Enable or highlight thinking checkbox
        self.show_thinking_checkbox.setEnabled(is_thinking_capable)
        if is_thinking_capable:
            self.show_thinking_checkbox.setStyleSheet("color: #007AFF; font-weight: bold;")
        else:
            self.show_thinking_checkbox.setStyleSheet("")
        
        # Update search status indicator
        self.update_search_status_indicator()

    def apply_settings(self):
        """Apply model settings for OpenAI"""
        try:
            # Update model settings
            current_model = self.model_selector.currentText()
            model_settings["model"] = current_model
            model_settings["temperature"] = float(self.temperature_entry.text())
            model_settings["max_tokens"] = int(self.max_tokens_entry.text())
            model_settings["show_thinking"] = self.show_thinking_checkbox.isChecked()
            
            # Only set top_p if the model supports it
            if supports_top_p(current_model):
                model_settings["top_p"] = float(self.top_p_entry.text())
            
            # Validate parameters
            if model_settings["temperature"] < 0 or model_settings["temperature"] > 2:
                raise ValueError("Temperature must be between 0 and 2")
            
            if "top_p" in model_settings and (model_settings["top_p"] < 0 or model_settings["top_p"] > 1):
                raise ValueError("Top P must be between 0 and 1")
                    
            if model_settings["max_tokens"] < 1:
                raise ValueError("Max tokens must be at least 1")
            
            # Update model indicator in status bar
            model_text = f"Model: {model_settings['model']}"
            if model_settings["show_thinking"]:
                model_text += " (Thinking)"
            self.model_indicator.setText(model_text)
            
            # Update parameter states to reflect new model selection
            self.update_parameter_states()
            
            QMessageBox.information(self, "Settings Updated", "OpenAI model settings updated successfully.")
            
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))

    def toggle_agent_mode(self, enabled):
        """Handle toggling agent mode on and off"""
        self.agent_enabled = enabled
        self.update_agent_status_indicator()
        self.update_page_header(self.chat_tabs.currentIndex())
        
        if enabled:
            # Enable or disable multi-agent controls based on multi-agent mode
            self.agent_count_spinner.setEnabled(self.multi_agent_enabled)
            for entry in self.agent_role_entries.values():
                entry.setEnabled(self.multi_agent_enabled)
            
            # When enabled, show confirmation and status
            current_page = self.chat_tabs.currentIndex()
            
            # Show a toast notification
            if self.multi_agent_enabled:
                Toast.show(self, f"Multi-agent dialog enabled with {self.agent_count_spinner.value()} agents")
            else:
                Toast.show(self, f"Agent mode enabled: {self.agent_name_entry.text()}")
        else:
            # Disable all agent controls
            self.agent_count_spinner.setEnabled(False)
            for entry in self.agent_role_entries.values():
                entry.setEnabled(False)
            
            # When disabled, inform the user
            Toast.show(self, "Agent mode disabled")
        
        # Update mode indicators in all tabs
        self.update_mode_indicators()
    
    def toggle_multi_agent_mode(self, enabled):
        """Handle toggling multi-agent mode on and off"""
        self.multi_agent_enabled = enabled
        
        # Enable or disable the agent count spinner
        self.agent_count_spinner.setEnabled(enabled and self.agent_enabled)
        
        # Enable or disable the role entries
        for entry in self.agent_role_entries.values():
            entry.setEnabled(enabled and self.agent_enabled)
        
        # Update the UI to reflect the new state
        self.update_agent_status_indicator()
        
        # Show notification
        Toast.show(
            self, 
            "Multi-agent dialog enabled - agents will converse sequentially" if enabled 
            else "Multi-agent dialog disabled"
        )
        
        # Update mode indicators in all tabs
        self.update_mode_indicators()
    
    def update_agent_status_indicator(self):
        """Update the visual indicator for agent status"""
        if self.agent_enabled:
            # Show multi-agent status if enabled
            if self.multi_agent_enabled:
                # Blue dot for multi-agent mode
                self.agent_status_indicator.setStyleSheet("color: #007AFF;")
                self.agent_status_label.setText(f"Multi-agent dialog active ({self.agent_count_spinner.value()} agents)")
                self.agent_mode_indicator.setText(f"Multi-Agent Mode ({self.agent_count_spinner.value()})")
                self.agent_mode_indicator.setStyleSheet("color: #007AFF;")
            else:
                # Green dot for single agent mode
                self.agent_status_indicator.setStyleSheet("color: #34C759;")
                self.agent_status_label.setText(f"Agent mode active: {self.agent_name_entry.text()}")
                self.agent_mode_indicator.setText("Agent Mode")
                self.agent_mode_indicator.setStyleSheet("color: #34C759;")
        else:
            # Gray dot for inactive
            self.agent_status_indicator.setStyleSheet("color: #8E8E93;")
            self.agent_status_label.setText("Agent mode inactive")
            self.agent_mode_indicator.setText("Standard Mode")
            self.agent_mode_indicator.setStyleSheet("color: #777777;")
    
    def update_page_header(self, page_index):
        """Update the page tab text to indicate agent mode if enabled"""
        if page_index < 0:
            return
            
        if self.agent_enabled:
            if self.multi_agent_enabled:
                self.chat_tabs.setTabText(page_index, f"Page {page_index+1} (Multi-Agent)")
            else:
                self.chat_tabs.setTabText(page_index, f"Page {page_index+1} (Agent)")
        else:
            self.chat_tabs.setTabText(page_index, f"Page {page_index+1}")
    
    def update_page_indicator(self, index):
        """Update the page indicator in the status bar"""
        self.page_indicator.setText(f"Page: {index+1}")
        self.update_page_header(index)
    
    def update_mode_indicators(self):
        """Show the current mode in each chat page interface"""
        # Loop through all chat tabs
        for page_idx in range(self.chat_tabs.count()):
            page = self.chat_tabs.widget(page_idx)
            
            # Find the actions frame (containing the Generate button)
            for i in range(page.layout().count()):
                widget = page.layout().itemAt(i).widget()
                if isinstance(widget, QFrame):
                    for j in range(widget.layout().count()):
                        item = widget.layout().itemAt(j)
                        if item.widget() and isinstance(item.widget(), QPushButton) and item.widget().text() == "Generate":
                            actions_frame = widget
                            actions_layout = actions_frame.layout()
                            
                            # Find and remove any existing mode indicators
                            for k in range(actions_layout.count()):
                                item_widget = actions_layout.itemAt(k).widget()
                                if isinstance(item_widget, QLabel) and "Mode" in item_widget.text():
                                    item_widget.deleteLater()
                            
                            # Create new mode indicator if needed
                            if self.agent_enabled:
                                # Create a new mode label
                                if self.multi_agent_enabled:
                                    mode_label = QLabel(f"Multi-Agent Mode ({self.agent_count_spinner.value()})")
                                    mode_label.setStyleSheet("color: #007AFF; font-style: italic;")
                                else:
                                    mode_label = QLabel("Agent Mode")
                                    mode_label.setStyleSheet("color: #34C759; font-style: italic;")
                                
                                mode_label.setFont(QFont("Segoe UI", 9))
                                
                                # Add to the end of the layout
                                # First add a stretch to push the label to the right
                                # Note: Need to check if a stretch already exists
                                has_stretch = False
                                for k in range(actions_layout.count()):
                                    if actions_layout.itemAt(k).spacerItem():
                                        has_stretch = True
                                        break
                                
                                if not has_stretch:
                                    actions_layout.addStretch()
                                
                                actions_layout.addWidget(mode_label)
    
    def apply_preset(self):
        """Apply a preset to the system instructions"""
        preset_text = {
            "General Assistant": "You are a helpful, harmless, and honest AI assistant.",
            "Code Assistant": "You are a programming assistant. Focus on providing accurate, secure code with explanations.",
            "Creative Writer": "You are a creative writing assistant. Be imaginative and engaging in your responses."
        }.get(self.preset_dropdown.currentText(), "")
        
        if preset_text:
            self.system_instructions_text.setText(preset_text)
            Toast.show(self, "Preset applied. Click 'Save Instructions' to keep changes.")
    
    def apply_agent_preset(self):
        """Apply a preset to the agent description"""
        preset_text = {
            "Research Assistant": "You are a research assistant that helps users find information and answers questions using the most up-to-date information available.",
            "Code Assistant": "You are a code assistant that helps users write, debug, and optimize code. You can search for coding solutions and documentation.",
            "Travel Planner": "You are a travel planning assistant that helps users plan trips, find accommodations, and discover attractions.",
            "Product Researcher": "You are a product research assistant that helps users compare products, find reviews, and make informed purchasing decisions."
        }.get(self.agent_preset_dropdown.currentText(), "")
        
        if preset_text:
            self.agent_description.setText(preset_text)
            self.agent_name_entry.setText(self.agent_preset_dropdown.currentText())
            Toast.show(self, f"Applied '{self.agent_preset_dropdown.currentText()}' preset")
    
    def reset_layout(self):
        """Reset layout to default settings"""
        global current_theme, themes, current_fonts
        
        # Reset themes to default
        themes = {
            "Dark": {
                "bg": "#2c3e50",
                "fg": "#ecf0f1",
                "input_bg": "#34495e",
                "output_bg": "#34495e",
                "accent": "#2980b9",
                "border": "#3d566e"
            },
            "Light": {
                "bg": "#f0f0f0",
                "fg": "#2c3e50",
                "input_bg": "#ffffff",
                "output_bg": "#ffffff",
                "accent": "#007AFF",
                "border": "#cccccc"
            },
            "Blue": {
                "bg": "#1e3d59",
                "fg": "#ecf0f1",
                "input_bg": "#3a506b",
                "output_bg": "#3a506b",
                "accent": "#0055D4",
                "border": "#4a6fa5"
            }
        }
        current_theme = "Dark"
        self.apply_theme(current_theme)
        
        # Reset fonts to default
        current_fonts = {
            "label": QFont("Segoe UI", 12, QFont.Bold),
            "input": QFont("Segoe UI", 11),
            "output": QFont("Segoe UI", 11),
            "heading": QFont("Segoe UI", 18, QFont.Bold),
            "button": QFont("Segoe UI", 11)
        }
        self.update_all_fonts()
        
        # Show toast
        Toast.show(self, "Layout reset to default", 1500)
    
    def load_user_preferences(self):
        """Load user preferences from a file"""
        global current_fonts, model_settings, current_theme
        
        try:
            with open("user_preferences.json", "r", encoding="utf-8") as file:
                preferences = json.load(file)
                
                # Load theme
                if "theme" in preferences:
                    self.apply_theme(preferences["theme"])
                
                # Load fonts
                if "fonts" in preferences:
                    for key, (family, size, bold, italic) in preferences["fonts"].items():
                        if key in current_fonts:
                            font = QFont(family, size)
                            font.setBold(bold)
                            font.setItalic(italic)
                            current_fonts[key] = font
                    self.update_all_fonts()
                
                # Load model settings
                if "model_settings" in preferences:
                    model_settings.update(preferences["model_settings"])
                    
                    # Update UI elements to reflect loaded settings
                    if hasattr(self, 'model_selector'):
                        index = self.model_selector.findText(model_settings["model"])
                        if index >= 0:
                            self.model_selector.setCurrentIndex(index)
                    
                    if hasattr(self, 'temperature_entry'):
                        self.temperature_entry.setText(str(model_settings.get("temperature", 0.7)))
                    
                    if hasattr(self, 'top_p_entry'):
                        self.top_p_entry.setText(str(model_settings.get("top_p", 1.0)))
                    
                    if hasattr(self, 'max_tokens_entry'):
                        self.max_tokens_entry.setText(str(model_settings.get("max_tokens", 1024)))
                        
                    if hasattr(self, 'show_thinking_checkbox'):
                        self.show_thinking_checkbox.setChecked(model_settings.get("show_thinking", False))
                
                # Load agent settings
                if "agent_enabled" in preferences:
                    self.agent_enabled = preferences["agent_enabled"]
                    if hasattr(self, 'agent_toggle'):
                        self.agent_toggle.setChecked(self.agent_enabled)
                
                if "multi_agent_enabled" in preferences:
                    self.multi_agent_enabled = preferences["multi_agent_enabled"]
                    if hasattr(self, 'multi_agent_toggle'):
                        self.multi_agent_toggle.setChecked(self.multi_agent_enabled)
                
                if "agent_roles" in preferences:
                    self.agent_roles = preferences["agent_roles"]
                    self.update_agent_roles_ui()
                
                # Load other UI preferences
                if "web_search_enabled" in preferences:
                    self.web_search_enabled = preferences["web_search_enabled"]
                    if hasattr(self, 'web_search_check'):
                        self.web_search_check.setChecked(self.web_search_enabled)
                
        except FileNotFoundError:
            # File doesn't exist yet, use defaults
            logging.info("User preferences file not found. Using defaults.")
        except Exception as e:
            logging.error(f"Failed to load user preferences: {e}")
    
    def save_user_preferences(self):
        """Save user preferences to a file"""
        preferences = {
            "theme": current_theme,
            "fonts": {key: (font.family(), font.pointSize(), font.bold(), font.italic()) 
                     for key, font in current_fonts.items()},
            "model_settings": model_settings,
            "agent_enabled": self.agent_enabled,
            "multi_agent_enabled": self.multi_agent_enabled,
            "web_search_enabled": self.web_search_enabled,
            "agent_roles": self.agent_roles
        }
        
        try:
            with open("user_preferences.json", "w", encoding="utf-8") as file:
                json.dump(preferences, file, indent=4)
        except Exception as e:
            logging.error(f"Failed to save user preferences: {e}")
    
    def open_gui_customizer(self):
        """Open the GUI customizer window"""
        customizer = QMainWindow(self)
        customizer.setWindowTitle("GUI Customizer")
        customizer.setMinimumSize(600, 500)
        
        # Central widget
        central = QWidget()
        customizer.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Theme name
        name_frame = QFrame()
        name_layout = QHBoxLayout(name_frame)
        name_layout.addWidget(QLabel("Theme Name:"))
        theme_name_entry = QLineEdit(current_theme)
        name_layout.addWidget(theme_name_entry)
        main_layout.addWidget(name_frame)
        
        # Color selectors
        color_frame = QFrame()
        color_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        color_layout = QVBoxLayout(color_frame)
        color_layout.addWidget(QLabel("Theme Colors"))
        
        # Create color pickers for each theme element
        color_pickers = {}
        theme = themes[current_theme]
        
        for color_key, color_name in [
            ("bg", "Background"),
            ("fg", "Text"),
            ("input_bg", "Input Background"), 
            ("output_bg", "Output Background"),
            ("accent", "Accent Color"),
            ("border", "Border Color")
        ]:
            row = QFrame()
            row_layout = QHBoxLayout(row)
            row_layout.addWidget(QLabel(f"{color_name}:"))
            
            color_button = QPushButton()
            color_button.setFixedSize(30, 30)
            color_button.setStyleSheet(f"background-color: {theme.get(color_key, '#FFFFFF')};")
            
            # Store reference to the button
            color_pickers[color_key] = color_button
            
            # Function to open color picker
            def pick_color(key=color_key, btn=color_button):
                current_color = QColor(theme.get(key, '#FFFFFF'))
                new_color = QColorDialog.getColor(current_color, customizer, f"Select {key} color")
                if new_color.isValid():
                    btn.setStyleSheet(f"background-color: {new_color.name()};")
                    # Update preview
                    update_preview()
            
            color_button.clicked.connect(pick_color)
            
            # Color value display
            color_value = QLineEdit(theme.get(color_key, '#FFFFFF'))
            color_value.setReadOnly(True)
            
            row_layout.addWidget(color_button)
            row_layout.addWidget(color_value)
            color_layout.addWidget(row)
        
        main_layout.addWidget(color_frame)
        
        # Preview
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.addWidget(QLabel("Preview"))
        
        preview_area = QFrame()
        preview_area.setMinimumHeight(150)
        preview_layout.addWidget(preview_area)
        
        # Sample elements in preview
        preview_input = QTextEdit("Sample input text")
        preview_input.setMaximumHeight(60)
        preview_layout.addWidget(preview_input)
        
        preview_button = QPushButton("Sample Button")
        preview_layout.addWidget(preview_button)
        
        main_layout.addWidget(preview_frame)
        
        # Function to update preview with current colors
        def update_preview():
            current_colors = {}
            for key, btn in color_pickers.items():
                color = btn.styleSheet().split(':')[1].strip().replace(';', '')
                current_colors[key] = color
            
            # Apply colors to preview
            preview_area.setStyleSheet(f"background-color: {current_colors.get('bg', '#FFFFFF')};")
            preview_input.setStyleSheet(f"""
                background-color: {current_colors.get('input_bg', '#FFFFFF')};
                color: {current_colors.get('fg', '#000000')};
                border: 1px solid {current_colors.get('border', '#CCCCCC')};
            """)
            preview_button.setStyleSheet(f"""
                background-color: {current_colors.get('accent', '#007AFF')};
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            """)
        
        # Initial preview update
        update_preview()
        
        # Action buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        
        save_theme_btn = QPushButton("Save Theme")
        
        def save_custom_theme():
            name = theme_name_entry.text().strip()
            if not name:
                QMessageBox.warning(customizer, "Input Error", "Please enter a theme name.")
                return
            
            # Collect colors from pickers
            new_theme = {}
            for key, btn in color_pickers.items():
                color = btn.styleSheet().split(':')[1].strip().replace(';', '')
                new_theme[key] = color
            
            # Add to themes dictionary
            themes[name] = new_theme
            
            # Apply the new theme
            self.apply_theme(name)
            
            # Add to theme selector if not already there
            if name not in self.theme_buttons:
                theme_radio = QCheckBox(name)
                theme_radio.setChecked(True)
                theme_radio.toggled.connect(lambda checked, tn=name: self.apply_theme(tn) if checked else None)
                self.theme_buttons[name] = theme_radio
                
                # Find the theme group layout in the appearance tab
                for i in range(self.settings_tabs.count()):
                    if self.settings_tabs.tabText(i) == "Appearance":
                        appearance_tab = self.settings_tabs.widget(i)
                        for j in range(appearance_tab.layout().count()):
                            widget = appearance_tab.layout().itemAt(j).widget()
                            if isinstance(widget, QFrame) and widget.layout() and widget.layout().count() > 0:
                                for k in range(widget.layout().count()):
                                    item = widget.layout().itemAt(k)
                                    if item.widget() and isinstance(item.widget(), QFrame):
                                        theme_group = item.widget()
                                        if theme_group.layout():
                                            theme_group.layout().addWidget(theme_radio)
                                            break
            
            # Save to file
            self.save_user_preferences()
            
            # Close customizer
            customizer.close()
            QMessageBox.information(self, "Theme Saved", f"Theme '{name}' has been saved and applied.")
        
        save_theme_btn.clicked.connect(save_custom_theme)
        button_layout.addWidget(save_theme_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(customizer.close)
        button_layout.addWidget(cancel_btn)
        
        main_layout.addWidget(button_frame)
        
        # Show the customizer dialog
        customizer.show()
    
    def load_env_file(self):
        """Load environment variables from a .env file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .env file", "", "Environment Files (*.env)")
        if file_path:
            load_dotenv(file_path)
            QMessageBox.information(self, "Environment Loaded", f"Environment variables loaded from {file_path}")
    
    def perform_file_operation(self, operation):
        """Perform file operations like load/save"""
        if operation == "load":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Text File", "", "Text Files (*.txt)")
            if file_path:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    self.input_entries[self.chat_tabs.currentIndex()].setPlainText(text)
                    QMessageBox.information(self, "File Loaded", f"Text loaded from {file_path}")
        elif operation == "save":
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Output to File", "", "Text Files (*.txt)")
            if file_path:
                with open(file_path, "w", encoding="utf-8") as file:
                    text = self.output_texts[self.chat_tabs.currentIndex()].toPlainText()
                    file.write(text)
                    QMessageBox.information(self, "File Saved", f"Output saved to {file_path}")
        elif operation == "save_session":
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Current Session", "", "Session Files (*.session)")
            if file_path:
                session_data = {
                    "input": [entry.toPlainText() for entry in self.input_entries],
                    "output": [text.toPlainText() for text in self.output_texts],
                    "settings": {
                        "theme": current_theme,
                        "fonts": {key: (font.family(), font.pointSize(), font.bold(), font.italic()) for key, font in current_fonts.items()},
                        "model_settings": model_settings,
                        "agent_enabled": self.agent_enabled,
                        "multi_agent_enabled": self.multi_agent_enabled,
                        "web_search_enabled": self.web_search_enabled,
                        "agent_roles": self.agent_roles
                    }
                }
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(session_data, file, indent=4)
                    QMessageBox.information(self, "Session Saved", f"Session saved to {file_path}")
        elif operation == "load_session":
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Saved Session", "", "Session Files (*.session)")
            if file_path:
                with open(file_path, "r", encoding="utf-8") as file:
                    session_data = json.load(file)
                    for i, text in enumerate(session_data["input"]):
                        self.input_entries[i].setPlainText(text)
                    for i, text in enumerate(session_data["output"]):
                        self.output_texts[i].setPlainText(text)
                    self.apply_theme(session_data["settings"]["theme"])
                    for key, (family, size, bold, italic) in session_data["settings"]["fonts"].items():
                        font = QFont(family, size)
                        font.setBold(bold)
                        font.setItalic(italic)
                        current_fonts[key] = font
                    self.update_all_fonts()
                    model_settings.update(session_data["settings"]["model_settings"])
                    self.agent_enabled = session_data["settings"]["agent_enabled"]
                    self.multi_agent_enabled = session_data["settings"]["multi_agent_enabled"]
                    self.web_search_enabled = session_data["settings"]["web_search_enabled"]
                    self.agent_roles = session_data["settings"]["agent_roles"]
                    self.update_agent_roles_ui()
                    self.update_mode_indicators()
                    self.update_agent_status_indicator()
                    self.update_page_header(self.chat_tabs.currentIndex())
                    QMessageBox.information(self, "Session Loaded", f"Session loaded from {file_path}")
    
    def generate_photo(self):
        """Generate an image based on the provided prompt using DALL-E"""
        global stop_event
        prompt = self.photo_prompt_entry.text()
        if not prompt:
            QMessageBox.warning(self, "Input Error", "Please enter a prompt for image generation.")
            return
        
        try:
            # Check for API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                QMessageBox.critical(self, "API Key Error", 
                    "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                return
            
            # Get selected model and size
            selected_model = self.image_model_selector.currentText()
            selected_size = self.image_size_selector.currentText().split('x')
            width, height = int(selected_size[0]), int(selected_size[1])
            
            # Display loading message
            self.status_left.setText("Generating image...")
            
            # Start progress animation
            self.progress_indicator.setStyleSheet(f"""
                background-color: {themes[current_theme].get("accent", "#007AFF")};
                border-radius: 4px;
                width: 0%;
            """)
            self.progress_bar.show()
            
            # Clear previous image
            self.image_display.clear()
            self.image_display.setText("Generating image...")
            self.save_image_button.setEnabled(False)
            
            # Create and run worker in a proper QThread
            self.img_qt_thread = QThread()
            self.img_worker = ImageGenerationWorker(prompt, selected_model, width, height)
            self.img_worker.moveToThread(self.img_qt_thread)
            
            # Connect signals
            self.img_qt_thread.started.connect(self.img_worker.generate)
            self.img_worker.generation_complete.connect(self.handle_image_generated)
            self.img_worker.generation_error.connect(self.handle_image_error)
            self.img_worker.generation_complete.connect(self.img_qt_thread.quit)
            self.img_worker.generation_error.connect(self.img_qt_thread.quit)
            self.img_qt_thread.finished.connect(self.img_qt_thread.deleteLater)
            
            # Start progress animation
            stop_event.clear()
            progress_thread = Thread(target=self.animate_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Start thread
            self.img_qt_thread.start()
            
        except Exception as e:
            logging.error(f"Image generation setup failed: {e}")
            QMessageBox.critical(self, "Generation Error", f"Failed to set up image generation: {str(e)}")

    def handle_image_generated(self, image):
        """Handle successful image generation"""
        global stop_event
        try:
            # Store the image for save functionality
            self.current_generated_image = image
            
            # Convert PIL Image to QPixmap for display - FIX CRASH ISSUE
            pixmap = None
            try:
                # Method 1: Convert using PIL to QImage conversion
                qim = ImageQt.ImageQt(image)
                pixmap = QPixmap.fromImage(qim)
            except Exception as e:
                logging.error(f"Primary image conversion failed: {e}")
                try:
                    # Method 2: Alternative conversion using temporary file
                    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_img.png")
                    image.save(temp_path)
                    pixmap = QPixmap(temp_path)
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e2:
                    logging.error(f"Secondary image conversion failed: {e2}")
                    # Create a fallback colored pixmap
                    pixmap = QPixmap(300, 200)
                    pixmap.fill(QColor("#cccccc"))
            
            if pixmap and not pixmap.isNull():
                # Scale the image to fit the display area while preserving aspect ratio
                display_width = self.image_display.width()
                display_height = self.image_display.height()
                scaled_pixmap = pixmap.scaled(display_width, display_height, Qt.KeepAspectRatio)
                
                # Update UI
                self.image_display.setPixmap(scaled_pixmap)
                self.save_image_button.setEnabled(True)
            else:
                # Fallback if pixmap creation failed
                self.image_display.setText("Image received but could not be displayed")
                self.save_image_button.setEnabled(True)  # Still allow saving as the PIL image is valid
            
            # Stop progress
            stop_event.set()
            self.progress_bar.hide()
            self.status_left.setText("Ready")
            
        except Exception as e:
            logging.error(f"Image handling failed: {e}")
            self.image_display.setText("Error processing image")
            self.save_image_button.setEnabled(False)
            # Stop progress
            stop_event.set()
            self.progress_bar.hide()
            self.status_left.setText("Error occurred")

    def handle_generation_error(self, error_message):
        """Handle errors during text generation"""
        global stop_event
        # Stop progress
        stop_event.set()
        self.progress_bar.hide()
        
        # Update status
        self.status_left.setText("Error occurred")
        
        # Show error in output text if there is a current page index
        if hasattr(self, 'current_page_index'):
            page_index = self.current_page_index
            current_text = self.output_texts[page_index].toPlainText()
            
            # Remove any "Processing..." or "Generating..." text
            if "Processing..." in current_text:
                current_text = current_text.rsplit("Processing...", 1)[0]
            if "Generating..." in current_text:
                current_text = current_text.rsplit("Generating...", 1)[0]
                
            # Add the error message
            self.output_texts[page_index].setPlainText(
                f"{current_text}\n\nError: {error_message}"
            )
        
        # Show error message
        logging.error(f"Generation error: {error_message}")
        QMessageBox.critical(self, "Generation Error", f"Failed to generate response: {error_message}")

    def handle_image_error(self, error_message):
        """Handle image generation error"""
        global stop_event
        self.image_display.setText("Error generating image")
        QMessageBox.critical(self, "Generation Error", f"Failed to generate image: {error_message}")
        
        # Stop progress
        stop_event.set()
        self.progress_bar.hide()
        self.status_left.setText("Error occurred")

    def save_generated_image(self):
        """Save the currently displayed generated image"""
        if not hasattr(self, 'current_generated_image') or not self.current_generated_image:
            QMessageBox.warning(self, "Save Error", "No image available to save.")
            return
        
        # Open file dialog to choose save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Generated Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                # Save the image
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path += '.png'  # Default to PNG if no extension
                
                self.current_generated_image.save(file_path)
                QMessageBox.information(self, "Image Saved", f"Image saved to {file_path}")
            except Exception as e:
                logging.error(f"Failed to save image: {e}")
                QMessageBox.critical(self, "Save Error", f"Failed to save image: {str(e)}")
    
    def update_goals(self):
        """Update AI goals"""
        QMessageBox.information(self, "Update AI Goals", "This feature is under development.")
    
    def view_goals(self):
        """View AI goals"""
        QMessageBox.information(self, "View AI Goals", "This feature is under development.")
    
    def load_instructions(self):
        """Load system and developer instructions from a file"""
        try:
            with open("instructions.json", "r", encoding="utf-8") as file:
                instructions = json.load(file)
                self.system_instructions_text.setPlainText(instructions.get("system", ""))
                self.developer_instructions_text.setPlainText(instructions.get("developer", ""))
        except FileNotFoundError:
            logging.warning("Instructions file not found. Using default instructions.")
        except Exception as e:
            logging.error(f"Failed to load instructions: {e}")
    
    def save_system_instructions(self):
        """Save system and developer instructions to unified settings file"""
        # Also save to the legacy file for backward compatibility
        instructions = {
            "system": self.system_instructions_text.toPlainText(),
            "developer": self.developer_instructions_text.toPlainText()
        }
        try:
            with open("instructions.json", "w", encoding="utf-8") as file:
                json.dump(instructions, file, indent=4)
            
            # Now save to unified settings
            success = save_app_settings(self)
            
            if success:
                QMessageBox.information(self, "Instructions Saved", "Instructions saved successfully.")
            else:
                QMessageBox.warning(self, "Partial Save", "Instructions saved to legacy file but failed to update unified settings.")
        except Exception as e:
            logging.error(f"Failed to save instructions: {e}")
            QMessageBox.critical(self, "Save Error", "Failed to save instructions. Please try again later.")
    
    def load_custom_actions(self):
        """Load custom actions from a file"""
        try:
            with open("custom_actions.json", "r", encoding="utf-8") as file:
                self.button_functions = json.load(file)
                for actions_menu in self.actions_menus:
                    actions_menu.clear()
                    actions_menu.addItem("Actions")
                    for action_name in self.button_functions.keys():
                        actions_menu.addItem(action_name)
        except FileNotFoundError:
            logging.warning("Custom actions file not found. Using default actions.")
        except Exception as e:
            logging.error(f"Failed to load custom actions: {e}")
    
    def open_button_manager(self):
        """Open the button manager window to create/edit custom actions with smart injection"""
        manager = QMainWindow(self)
        manager.setWindowTitle("Button Manager")
        manager.setMinimumSize(700, 500)
        
        # Central widget
        central = QWidget()
        manager.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Actions list
        actions_frame = QFrame()
        actions_layout = QHBoxLayout(actions_frame)
        
        # Left side - actions list
        list_frame = QFrame()
        list_layout = QVBoxLayout(list_frame)
        
        list_layout.addWidget(QLabel("Available Actions:"))
        
        actions_list = QListWidget()
        for action_name in self.button_functions.keys():
            actions_list.addItem(action_name)
        
        list_layout.addWidget(actions_list)
        actions_layout.addWidget(list_frame)
        
        # Right side - action editor
        editor_frame = QFrame()
        editor_layout = QVBoxLayout(editor_frame)
        
        editor_layout.addWidget(QLabel("Action Details:"))
        
        # Action name
        name_frame = QFrame()
        name_layout = QHBoxLayout(name_frame)
        name_layout.addWidget(QLabel("Name:"))
        action_name_edit = QLineEdit()
        name_layout.addWidget(action_name_edit)
        editor_layout.addWidget(name_frame)
        
        # Action type
        type_frame = QFrame()
        type_layout = QHBoxLayout(type_frame)
        type_layout.addWidget(QLabel("Type:"))
        action_type_combo = QComboBox()
        action_type_combo.addItems(["insert_text", "generate", "clear", "execute_code"])
        type_layout.addWidget(action_type_combo)
        editor_layout.addWidget(type_frame)
        
        # Action text
        text_frame = QFrame()
        text_layout = QVBoxLayout(text_frame)
        text_layout.addWidget(QLabel("Text to Insert:"))
        action_text_edit = QTextEdit()
        action_text_edit.setPlaceholderText("Enter text to insert or code to execute")
        action_text_edit.setMinimumHeight(150)
        text_layout.addWidget(action_text_edit)
        editor_layout.addWidget(text_frame)
        
        # Action options
        options_frame = QFrame()
        options_layout = QVBoxLayout(options_frame)
        
        replace_check = QCheckBox("Replace current text")
        options_layout.addWidget(replace_check)
        
        # Add smart inject option
        smart_inject_check = QCheckBox("Smart inject (transform existing content)")
        smart_inject_check.setToolTip(
            "When enabled, the action text is treated as instructions to apply to the existing content.\n"
            "For example: 'Refactor this code using best practices' will be applied to the current input."
        )
        options_layout.addWidget(smart_inject_check)
        
        # Add some formatting help
        format_frame = QFrame()
        format_layout = QVBoxLayout(format_frame)
        format_layout.addWidget(QLabel("Common Smart Inject Templates:"))
        
        templates_layout = QVBoxLayout()
        templates = [
            ("Code Refactor", "Please refactor the following code to improve readability, efficiency, and follow best practices:"),
            ("Summarize", "Please summarize the following content in a concise manner:"),
            ("Expand", "Please expand on the following content with more details and examples:"),
            ("Fix Grammar", "Please correct any grammar, spelling, or punctuation errors in the following text:"),
            ("Convert to API", "Please convert the following code into a proper API with endpoints and documentation:")
        ]
        
        for name, template in templates:
            template_btn = QPushButton(name)
            template_btn.clicked.connect(lambda _, t=template: action_text_edit.setPlainText(t))
            templates_layout.addWidget(template_btn)
        
        format_frame.setLayout(templates_layout)
        options_layout.addWidget(format_frame)
        
        target_frame = QFrame()
        target_layout = QHBoxLayout(target_frame)
        target_layout.addWidget(QLabel("Target (for Clear action):"))
        
        target_combo = QComboBox()
        target_combo.addItems(["input", "output", "both"])
        target_layout.addWidget(target_combo)
        options_layout.addWidget(target_frame)
        
        editor_layout.addWidget(options_frame)
        
        # Buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        
        save_btn = QPushButton("Save")
        new_btn = QPushButton("New")
        delete_btn = QPushButton("Delete")
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(new_btn)
        button_layout.addWidget(delete_btn)
        
        editor_layout.addWidget(button_frame)
        editor_layout.addStretch()
        
        actions_layout.addWidget(editor_frame)
        main_layout.addWidget(actions_frame)
        
        # Close button at bottom
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(manager.close)
        main_layout.addWidget(close_btn)
        
        # Function to load selected action
        def load_action(item):
            action_name = item.text()
            action_def = self.button_functions.get(action_name, {})
            
            action_name_edit.setText(action_name)
            
            # Set action type
            action_type = action_def.get("type", "insert_text")
            action_type_index = action_type_combo.findText(action_type)
            if action_type_index >= 0:
                action_type_combo.setCurrentIndex(action_type_index)
            
            # Set action text or code
            if action_type == "execute_code":
                action_text_edit.setPlainText(action_def.get("code", ""))
            else:
                action_text_edit.setPlainText(action_def.get("text", ""))
            
            # Set replace option
            replace_check.setChecked(action_def.get("replace", False))
            
            # Set smart inject option
            smart_inject_check.setChecked(action_def.get("smart_inject", False))
            
            # Set target option
            target_index = target_combo.findText(action_def.get("target", "both"))
            if target_index >= 0:
                target_combo.setCurrentIndex(target_index)
        
        # Function to save the current action
        def save_action():
            action_name = action_name_edit.text().strip()
            if not action_name:
                QMessageBox.warning(manager, "Input Error", "Please enter an action name.")
                return
            
            # Create action definition
            action_type = action_type_combo.currentText()
            action_def = {"type": action_type}
            
            if action_type == "execute_code":
                action_def["code"] = action_text_edit.toPlainText()
            else:
                action_def["text"] = action_text_edit.toPlainText()
            
            if action_type in ["insert_text", "generate"]:
                action_def["replace"] = replace_check.isChecked()
                action_def["smart_inject"] = smart_inject_check.isChecked()
            
            if action_type == "clear":
                action_def["target"] = target_combo.currentText()
            
            # Save to button functions dictionary
            self.button_functions[action_name] = action_def
            
            # Update actions list if needed
            found = False
            for i in range(actions_list.count()):
                if actions_list.item(i).text() == action_name:
                    found = True
                    break
            
            if not found:
                actions_list.addItem(action_name)
            
            # Save to file
            with open("custom_actions.json", "w", encoding="utf-8") as file:
                json.dump(self.button_functions, file, indent=4)
            
            # Update all action menus
            for actions_menu in self.actions_menus:
                current_selection = actions_menu.currentText()
                actions_menu.clear()
                actions_menu.addItem("Actions")
                for action_name in self.button_functions.keys():
                    actions_menu.addItem(action_name)
                
                # Try to restore previous selection
                index = actions_menu.findText(current_selection)
                if index >= 0:
                    actions_menu.setCurrentIndex(index)
            
            # Show confirmation
            Toast.show(manager, f"Action '{action_name}' saved", 1500)
        
        # Function to create a new action
        def new_action():
            action_name_edit.clear()
            action_type_combo.setCurrentIndex(0)
            action_text_edit.clear()
            replace_check.setChecked(False)
            smart_inject_check.setChecked(False)
            target_combo.setCurrentIndex(2)  # "both"
        
        # Function to delete the current action
        def delete_action():
            action_name = action_name_edit.text().strip()
            if not action_name:
                return
            
            # Confirm deletion
            reply = QMessageBox.question(manager, "Confirm Deletion", 
                                        f"Are you sure you want to delete action '{action_name}'?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            
            # Delete from dictionary
            if action_name in self.button_functions:
                del self.button_functions[action_name]
            
            # Delete from list widget
            items = actions_list.findItems(action_name, Qt.MatchExactly)
            for item in items:
                row = actions_list.row(item)
                actions_list.takeItem(row)
            
            # Save to file
            with open("custom_actions.json", "w", encoding="utf-8") as file:
                json.dump(self.button_functions, file, indent=4)
            
            # Update all action menus
            for actions_menu in self.actions_menus:
                actions_menu.clear()
                actions_menu.addItem("Actions")
                for action_name in self.button_functions.keys():
                    actions_menu.addItem(action_name)
            
            # Clear the editor
            new_action()
            
            # Show confirmation
            Toast.show(manager, f"Action '{action_name}' deleted", 1500)
        
        # Function to update UI based on selected action type
        def update_ui_for_action_type(index):
            action_type = action_type_combo.currentText()
            
            # Enable/disable options based on action type
            smart_inject_check.setEnabled(action_type in ["insert_text", "generate"])
            replace_check.setEnabled(action_type in ["insert_text", "generate"])
            target_combo.setEnabled(action_type == "clear")
            
            # Update placeholder text
            if action_type == "execute_code":
                action_text_edit.setPlaceholderText("Enter Python code to execute")
            else:
                action_text_edit.setPlaceholderText("Enter text to insert or instructions")
        
        # Connect signals
        actions_list.itemClicked.connect(load_action)
        save_btn.clicked.connect(save_action)
        new_btn.clicked.connect(new_action)
        delete_btn.clicked.connect(delete_action)
        action_type_combo.currentIndexChanged.connect(update_ui_for_action_type)
        
        # Initial UI update
        update_ui_for_action_type(0)
        
        # Show the manager window
        manager.show()
    
    def execute_action(self, menu, page_index):
        """Execute the selected custom action with smart injection capability"""
        action_name = menu.currentText()
        
        # Reset to the default selection after execution
        menu.setCurrentIndex(0)
        
        # Skip if the default item is selected
        if (action_name == "Actions"):
            return
        
        # Get the action definition
        action_def = self.button_functions.get(action_name)
        if not action_def:
            QMessageBox.warning(self, "Action Error", f"No definition found for action '{action_name}'.")
            return
        
        try:
            # Get the current input text
            current_text = self.input_entries[page_index].toPlainText()
            
            # Handle different action types
            action_type = action_def.get("type", "insert_text")
            
            if action_type == "insert_text":
                # Insert predefined text
                text_to_insert = action_def.get("text", "")
                
                # Replace placeholders if any
                if "{DATE}" in text_to_insert:
                    from datetime import datetime
                    text_to_insert = text_to_insert.replace("{DATE}", datetime.now().strftime("%Y-%m-%d"))
                if "{TIME}" in text_to_insert:
                    from datetime import datetime
                    text_to_insert = text_to_insert.replace("{TIME}", datetime.now().strftime("%H:%M:%S"))
                
                # Check for smart injection mode
                if action_def.get("smart_inject", False):
                    # Format as instructions for the existing content
                    if current_text.strip():
                        # Create an instruction that references the current content
                        final_text = f"{text_to_insert}\n\nContent to process:\n```\n{current_text}\n```"
                        self.input_entries[page_index].setPlainText(final_text)
                    else:
                        # Just insert the text if no content to process
                        self.input_entries[page_index].setPlainText(text_to_insert)
                else:
                    # Normal text insertion (previous behavior)
                    if action_def.get("replace", False):
                        self.input_entries[page_index].setPlainText(text_to_insert)
                    else:
                        # Insert at cursor position or append
                        cursor = self.input_entries[page_index].textCursor()
                        if cursor.hasSelection():
                            cursor.insertText(text_to_insert)
                        else:
                            if current_text and not current_text.endswith(("\n", " ")):
                                text_to_insert = " " + text_to_insert
                            self.input_entries[page_index].setPlainText(current_text + text_to_insert)
                
            elif action_type == "generate":
                # Auto-trigger generation after inserting text
                text_to_insert = action_def.get("text", "")
                
                # Check for smart injection mode
                if action_def.get("smart_inject", False) and current_text.strip():
                    # Format as instructions for the existing content
                    final_text = f"{text_to_insert}\n\nContent to process:\n```\n{current_text}\n```"
                    self.input_entries[page_index].setPlainText(final_text)
                else:
                    # Normal text insertion
                    if action_def.get("replace", False):
                        self.input_entries[page_index].setPlainText(text_to_insert)
                    else:
                        if current_text and not current_text.endswith(("\n", " ")):
                            text_to_insert = " " + text_to_insert
                        self.input_entries[page_index].setPlainText(current_text + text_to_insert)
                
                # Trigger generation after a short delay
                QTimer.singleShot(100, lambda: self.generate_response(page_index))
            
            elif action_type == "clear":
                # Clear the input or output or both
                target = action_def.get("target", "both")
                if target in ["input", "both"]:
                    self.input_entries[page_index].clear()
                if target in ["output", "both"]:
                    self.output_texts[page_index].clear()
            
            elif action_type == "execute_code":
                # Execute custom Python code
                code = action_def.get("code", "")
                if code:
                    # Prepare local variables that the code can use
                    local_vars = {
                        "app": self,
                        "page_index": page_index,
                        "input_text": current_text,
                        "output_text": self.output_texts[page_index].toPlainText()
                    }
                    
                    # Execute the code with the prepared local variables
                    exec(code, {}, local_vars)
                    
                    # Update the UI if the code modified the local variables
                    if "input_text" in local_vars and local_vars["input_text"] != current_text:
                        self.input_entries[page_index].setPlainText(local_vars["input_text"])
                    if "output_text" in local_vars and local_vars["output_text"] != self.output_texts[page_index].toPlainText():
                        self.output_texts[page_index].setPlainText(local_vars["output_text"])
                
            # Show toast notification
            Toast.show(self, f"Action '{action_name}' executed", 1500)
            
        except Exception as e:
            logging.error(f"Failed to execute action '{action_name}': {e}")
            QMessageBox.critical(self, "Action Error", f"Failed to execute action: {str(e)}")
    
    def generate_response(self, page_index):
        """Generate a response based on the input text using OpenAI API with streaming"""
        global stop_event
        # Check if agent mode is enabled
        if self.agent_enabled:
            self.run_agent(self.input_entries[page_index].toPlainText(), page_index)
            return
            
        # Regular generation process if not in agent mode
        input_text = self.input_entries[page_index].toPlainText()
        if not input_text:
            QMessageBox.warning(self, "Input Error", "Please enter some text to generate a response.")
            return
        
        # Log the user input to file
        log_conversation(page_index, "User", input_text)
        
        # Get system and developer instructions if they exist
        system_instructions = ""
        developer_instructions = ""
        
        if hasattr(self, 'system_instructions_text'):
            system_instructions = self.system_instructions_text.toPlainText().strip()
        
        if hasattr(self, 'developer_instructions_text'):
            developer_instructions = self.developer_instructions_text.toPlainText().strip()
        
        # Initialize chat history for this page if it doesn't exist
        if page_index not in chat_histories:
            chat_histories[page_index] = []
        
        # Add the new user message to history
        chat_histories[page_index].append({"role": "user", "content": input_text})
        
        # Prepare messages array for OpenAI API
        messages = []
        
        # Add system message if available and model supports it
        if system_instructions and supports_system_instructions(model_settings["model"]):
            messages.append({"role": "system", "content": system_instructions})
        
        # Add developer instructions if it's an O-family model
        if developer_instructions and supports_developer_instructions(model_settings["model"]):
            if messages and messages[0]["role"] == "system":
                # Add to existing system message
                messages[0]["content"] += f"\n\nDeveloper instructions: {developer_instructions}"
            else:
                # Create new system message
                messages.append({"role": "system", "content": f"Developer instructions: {developer_instructions}"})
        
        # Add conversation history
        for msg in chat_histories[page_index]:
            messages.append(msg)
        
        # Show conversation history in output box
        conversation_display = ""
        for msg in chat_histories[page_index]:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            conversation_display += f"{prefix}{msg['content']}\n\n"
            
        self.output_texts[page_index].setPlainText(conversation_display + "Assistant: ")
        
        # Create a text cursor at the end
        cursor = self.output_texts[page_index].textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_texts[page_index].setTextCursor(cursor)
        
        # Show progress bar
        self.progress_indicator.setStyleSheet(f"""
            background-color: {themes[current_theme].get("accent", "#007AFF")};
            border-radius: 4px;
            width: 0%;
        """)
        self.progress_bar.show()
        
        # Update status
        self.status_left.setText("Generating response...")
        
        # Start progress animation
        stop_event.clear()
        progress_thread = Thread(target=self.animate_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Create and run streaming worker in a thread
        self.qt_thread = QThread()
        self.streaming_worker = OpenAIStreamingWorker(json.dumps(messages), model_settings)
        self.streaming_worker.moveToThread(self.qt_thread)
        
        # Connect signals
        self.qt_thread.started.connect(self.streaming_worker.generate)
        self.streaming_worker.token_received.connect(lambda token: self.handle_streaming_token(token, page_index))
        self.streaming_worker.generation_complete.connect(lambda: self.handle_streaming_complete(page_index))
        self.streaming_worker.generation_error.connect(self.handle_generation_error)
        self.streaming_worker.generation_complete.connect(self.qt_thread.quit)
        self.streaming_worker.generation_error.connect(self.qt_thread.quit)
        self.qt_thread.finished.connect(self.qt_thread.deleteLater)
        
        # Store current page index for the handler
        self.current_page_index = page_index
        
        # Start thread
        self.qt_thread.start()

    def run_agent(self, input_text, page_index):
        """Run agent mode for response generation"""
        global stop_event
        if not input_text:
            QMessageBox.warning(self, "Input Error", "Please enter some text for the agent to process.")
            return
        
        # Log the user input for agent processing
        log_conversation(page_index, "User", input_text)
        
        # Update output to show processing status
        current_output = self.output_texts[page_index].toPlainText()
        agent_prefix = "Multi-Agent Dialog" if self.multi_agent_enabled else f"Agent ({self.agent_name_entry.text()})"
        processing_message = f"\n\n{agent_prefix}: Processing..."
        
        # Show conversation and processing message
        if current_output:
            display_text = current_output + processing_message
        else:
            display_text = f"User: {input_text}\n{processing_message}"
        
        self.output_texts[page_index].setPlainText(display_text)
        
        # Show progress bar
        self.progress_indicator.setStyleSheet(f"""
            background-color: {themes[current_theme].get("accent", "#007AFF")};
            border-radius: 4px;
            width: 0%;
        """)
        self.progress_bar.show()
        
        # Update status
        agent_mode = "Multi-Agent Dialog" if self.multi_agent_enabled else "Agent Mode"
        self.status_left.setText(f"Running {agent_mode}...")
        
        # Determine which agent process to run
        if self.multi_agent_enabled:
            # Multi-agent dialog process
            self.run_multi_agent_dialog(input_text, page_index)
        else:
            # Single agent process
            self.run_single_agent(input_text, page_index)

    def run_single_agent(self, input_text, page_index):
         """Run a single agent to process the input with streaming"""
         global stop_event
         try:
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                QMessageBox.critical(self, "API Key Error", 
                    "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                return
            
            # Get agent settings
            agent_name = self.agent_name_entry.text()
            agent_instructions = self.agent_description.toPlainText()
            agent_model = self.agent_model_selector.currentText()
            
            # Get agent memory for this page
            agent_memory_content = ""
            if hasattr(self, 'agent_memory') and page_index in self.active_agents:
                agent_id = self.active_agents[page_index]
                agent_memory_content = self.agent_memory.summarize_memories(agent_id)
                if agent_memory_content:
                    agent_memory_content = f"\nYour memory contains the following information:\n{agent_memory_content}"
            
            # Update output to show processing status
            current_output = self.output_texts[page_index].toPlainText()
            
            # Format the conversation
            if current_output.strip():
                display_text = f"{current_output}\n\nUser: {input_text}\n\n{agent_name}: "
            else:
                display_text = f"User: {input_text}\n\n{agent_name}: "
            
            self.output_texts[page_index].setPlainText(display_text)
            
            # Create a text cursor at the end
            cursor = self.output_texts[page_index].textCursor()
            cursor.movePosition(QTextCursor.End)
            self.output_texts[page_index].setTextCursor(cursor)
            
            # Show progress bar
            self.progress_indicator.setStyleSheet(f"""
                background-color: {themes[current_theme].get("accent", "#007AFF")};
                border-radius: 4px;
                width: 0%;
            """)
            self.progress_bar.show()
            
            # Update status
            self.status_left.setText("Agent processing...")
            
            # Start progress animation
            stop_event.clear()
            progress_thread = Thread(target=self.animate_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Create and run agent streaming worker in a thread
            self.qt_thread = QThread()
            self.agent_streaming_worker = AgentStreamingWorker(
                input_text, 
                agent_name, 
                agent_instructions, 
                agent_model,
                agent_memory_content
            )
            self.agent_streaming_worker.moveToThread(self.qt_thread)
            
            # Connect signals
            self.qt_thread.started.connect(self.agent_streaming_worker.generate)
            self.agent_streaming_worker.token_received.connect(lambda token: self.handle_streaming_token(token, page_index))
            self.agent_streaming_worker.generation_complete.connect(lambda: self.handle_agent_streaming_complete(page_index))
            self.agent_streaming_worker.generation_error.connect(self.handle_generation_error)
            self.agent_streaming_worker.generation_complete.connect(self.qt_thread.quit)
            self.agent_streaming_worker.generation_error.connect(self.qt_thread.quit)
            self.qt_thread.finished.connect(self.qt_thread.deleteLater)
            
            # Store current page index for the handler
            self.current_page_index = page_index
            
            # Start thread
            self.qt_thread.start()
            
         except Exception as e:
            logging.error(f"Agent generation failed: {e}")
            self.handle_generation_error(str(e))

    def run_multi_agent_dialog(self, input_text, page_index):
        """Run a multi-agent dialog to process the input"""
        global stop_event
        try:
            # Get the number of agents and their roles
            num_agents = self.agent_count_spinner.value()
            agent_model = self.agent_model_selector.currentText()
            
            # Get continuous mode settings
            interaction_mode = self.interaction_mode_selector.currentText()
            continuous_mode = (interaction_mode == "Continuous Debate")
            max_turns = None if self.turn_limit_spinner.value() == 0 else self.turn_limit_spinner.value()
            
            # Update prompt based on interaction mode to encourage discussion
            enhanced_prompt = input_text
            if interaction_mode == "Interactive" or interaction_mode == "Continuous Debate":
                # Add instructions for more interactive discussion
                enhanced_prompt = (
                    f"{input_text}\n\n"
                    f"This question will be discussed by {num_agents} AI agents with different perspectives. "
                    f"Each agent should not only answer the question but also engage with what previous agents have said."
                )
            
            # Create and run dialog worker in a thread
            self.dialog_qt_thread = QThread()
            self.dialog_worker = OpenAIDialogWorker(
                enhanced_prompt, 
                self.agent_roles,
                num_agents,
                agent_model,
                continuous_mode=continuous_mode,
                max_turns=max_turns
            )
            self.dialog_worker.moveToThread(self.dialog_qt_thread)
            
            # Connect signals
            self.dialog_qt_thread.started.connect(self.dialog_worker.generate_dialog)
            self.dialog_worker.agent_response.connect(self.handle_dialog_response)
            self.dialog_worker.dialog_complete.connect(self.handle_dialog_complete)
            self.dialog_worker.dialog_error.connect(self.handle_generation_error)
            self.dialog_worker.dialog_complete.connect(self.dialog_qt_thread.quit)
            self.dialog_worker.dialog_error.connect(self.dialog_qt_thread.quit)
            self.dialog_qt_thread.finished.connect(self.dialog_qt_thread.deleteLater)
            
            # Store current page index and worker for the handler
            self.current_page_index = page_index
            
            # Start progress animation
            stop_event.clear()
            progress_thread = Thread(target=self.animate_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Start thread
            self.dialog_qt_thread.start()
            
        except Exception as e:
            logging.error(f"Multi-agent dialog generation failed: {e}")
            self.handle_generation_error(str(e))

    def handle_agent_response(self, response_text):
        """Handle the response from a single agent"""
        global stop_event
        # Stop progress
        stop_event.set()
        self.progress_bar.hide()
        
        # Update output text with agent response
        page_index = self.current_page_index
        agent_name = self.agent_name_entry.text()
        
        # Log the agent response
        log_conversation(page_index, f"Agent ({agent_name})", response_text)
        
        # Format the conversation
        if self.output_texts[page_index].toPlainText().strip().endswith("Processing..."):
            # Remove the "Processing..." text
            current_text = self.output_texts[page_index].toPlainText()
            current_text = current_text.rsplit("Processing...", 1)[0]
            display_text = f"{current_text}{agent_name}: {response_text}"
        else:
            # Start fresh conversation
            display_text = f"User: {self.input_entries[page_index].toPlainText()}\n\n{agent_name}: {response_text}"
        
        self.output_texts[page_index].setPlainText(display_text)
        
        # Store in agent memory
        if hasattr(self, 'agent_memory'):
            # Create agent ID for this page if it doesn't exist
            if page_index not in self.active_agents:
                self.active_agents[page_index] = f"agent_{page_index}"
            
            # Extract key information from response for memory
            memory_text = f"User asked: {self.input_entries[page_index].toPlainText()[:50]}... You responded about: {response_text[:100]}..."
            self.agent_memory.add_memory(self.active_agents[page_index], memory_text)
        
        # Update status
        self.status_left.setText("Ready")

    def handle_dialog_response(self, agent_index, response_text):
        """Handle response from an agent in multi-agent dialog"""
        page_index = self.current_page_index
        
        # Log the agent response
        log_conversation(page_index, f"Agent {agent_index+1}", response_text)
        
        # Update output text to show each agent's response
        current_text = self.output_texts[page_index].toPlainText()
        
        # Format the agent's name based on role if available
        agent_role_snippet = ""
        if hasattr(self, 'agent_roles') and agent_index in self.agent_roles:
            # Extract a short descriptor from the role
            role_text = self.agent_roles[agent_index]
            # Get first sentence or phrase up to 40 chars
            short_role = role_text.split('.')[0][:40].strip()
            if short_role:
                agent_role_snippet = f" ({short_role})"
        
        # For the first agent, replace the "Processing..." text
        if agent_index == 0 and current_text.strip().endswith("Processing..."):
            current_text = current_text.rsplit("Processing...", 1)[0]
            display_text = f"{current_text}\n\nAgent {agent_index+1}{agent_role_snippet}:\n{response_text}\n"
        else:
            # For subsequent agents, append to previous responses
            display_text = f"{current_text}\n\nAgent {agent_index+1}{agent_role_snippet}:\n{response_text}\n"
        
        self.output_texts[page_index].setPlainText(display_text)
        
        # Scroll to bottom
        cursor = self.output_texts[page_index].textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_texts[page_index].setTextCursor(cursor)

    def handle_dialog_complete(self):
        """Handle completion of multi-agent dialog"""
        global stop_event
        # Stop progress
        stop_event.set()
        self.progress_bar.hide()
        
        # Update status
        self.status_left.setText("Ready")
        
        # Show completion message
        Toast.show(self, "Multi-agent dialog complete", 1500)

    def animate_progress(self):
        """Animate the progress bar during generation"""
        width = 0
        increment = 3
        progress_bar_width = self.progress_bar.width()
        
        while not stop_event.is_set() and width < progress_bar_width:
            width += increment
            if width >= progress_bar_width:
                width = 0  # Reset the animation
            
            # Update UI in thread-safe way
            QApplication.processEvents()
            self.progress_indicator.setStyleSheet(f"""
                background-color: {themes[current_theme].get("accent", "#007AFF")};
                border-radius: 4px;
                width: {width}px;
            """)
            self.progress_indicator.setFixedWidth(width)
            
            # Slow down the animation
            time.sleep(0.03)
        
        # Hide progress bar when done
        if stop_event.is_set():
            self.progress_bar.hide()

    def handle_streaming_token(self, token, page_index):
        """Handle each token as it arrives from the streaming response"""
        # Get current text
        cursor = self.output_texts[page_index].textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Insert the token at cursor position
        cursor.insertText(token)
        
        # Keep cursor at the end
        self.output_texts[page_index].setTextCursor(cursor)
        
        # Process events to update UI
        QApplication.processEvents()

    def handle_streaming_complete(self, page_index):
        """Handle completion of streaming response"""
        global stop_event
        # Stop progress
        stop_event.set()
        self.progress_bar.hide()
        
        # Update status
        self.status_left.setText("Ready")
        
        # Get the complete generated text (minus the "Assistant: " prefix)
        current_text = self.output_texts[page_index].toPlainText()
        response_text = current_text.split("Assistant: ")[-1]
        
        # Add the response to chat history
        if page_index not in chat_histories:
            chat_histories[page_index] = []
        
        chat_histories[page_index].append({"role": "assistant", "content": response_text})
        
        # Log the response to file
        log_conversation(page_index, "Assistant", response_text)
        
        # Add a newline for readability
        cursor = self.output_texts[page_index].textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n\n")
        self.output_texts[page_index].setTextCursor(cursor)

    def handle_standard_response(self, response_text, page_index):
        """Handle the response from standard (non-agent) generation"""
        global stop_event
        # Stop progress
        stop_event.set()
        self.progress_bar.hide()
        
        # Update status
        self.status_left.setText("Ready")
        
        # Add the response to chat history
        if page_index not in chat_histories:
            chat_histories[page_index] = []
            
        chat_histories[page_index].append({"role": "assistant", "content": response_text})
        
        # Log the response to file
        log_conversation(page_index, "Assistant", response_text)
        
        # Build the full conversation display
        conversation_display = ""
        for msg in chat_histories[page_index]:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            conversation_display += f"{prefix}{msg['content']}\n\n"
        
        # Remove trailing "Generating..." text if present
        if conversation_display.endswith("Assistant: Generating...\n\n"):
            conversation_display = conversation_display.replace("Assistant: Generating...\n\n", "")
        
        # Update output text
        self.output_texts[page_index].setPlainText(conversation_display)
        
        # Scroll to bottom
        cursor = self.output_texts[page_index].textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_texts[page_index].setTextCursor(cursor)

    def handle_agent_streaming_complete(self, page_index):
        """Handle completion of agent streaming response"""
        global stop_event
        # Stop progress
        stop_event.set()
        self.progress_bar.hide()
        
        # Update status
        self.status_left.setText("Ready")
        
        # Get the agent name
        agent_name = self.agent_name_entry.text()
        
        # Extract the full response for logging and memory
        current_text = self.output_texts[page_index].toPlainText()
        response_text = current_text.split(f"{agent_name}: ")[-1]
        
        # Log the agent response
        log_conversation(page_index, f"Agent ({agent_name})", response_text)
        
        # Store in agent memory
        if hasattr(self, 'agent_memory'):
            # Create agent ID for this page if it doesn't exist
            if page_index not in self.active_agents:
                self.active_agents[page_index] = f"agent_{page_index}"
            
            # Extract key information from response for memory
            memory_text = f"User asked: {self.input_entries[page_index].toPlainText()[:50]}... You responded about: {response_text[:100]}..."
            self.agent_memory.add_memory(self.active_agents[page_index], memory_text)
        
        # Add a newline for readability
        cursor = self.output_texts[page_index].textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n\n")
        self.output_texts[page_index].setTextCursor(cursor)

    def clear_history(self, page_index):
        """Clear conversation history for a specific chat tab"""
        # Clear the history in memory
        if page_index in chat_histories:
            chat_histories[page_index] = []
        
        # Clear the input and output fields
        self.input_entries[page_index].clear()
        self.output_texts[page_index].clear()
        
        # Show confirmation
        Toast.show(self, f"History cleared for Page {page_index+1}", 1500)

# Add a worker class for multi-agent dialog using OpenAI
class OpenAIDialogWorker(QObject):
    agent_response = pyqtSignal(int, str)
    dialog_complete = pyqtSignal()
    dialog_error = pyqtSignal(str)
    
    def __init__(self, prompt, agent_roles, num_agents, model_name, continuous_mode=False, max_turns=None):
        super().__init__()
        self.prompt = prompt
        self.agent_roles = agent_roles
        self.num_agents = num_agents
        self.model_name = model_name
        self.conversation_history = []
        self.continuous_mode = continuous_mode
        self.max_turns = max_turns
        self.current_turn = 0
        self.stop_requested = False
        
    def request_stop(self):
        """Request the dialog to stop after current agent completes"""
        self.stop_requested = True
    
    @pyqtSlot()
    def generate_dialog(self):
        """Generate a conversation between multiple agents using OpenAI"""
        try:
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                self.dialog_error.emit("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                return
            
            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Add user query to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": self.prompt
            })
            
            # Run initial agent responses
            self.current_turn = 0
            agent_idx = 0
            
            # Continue until stopped or max turns reached
            while not self.stop_requested and (self.max_turns is None or self.current_turn < self.max_turns):
                # Get agent role description
                agent_role = self.agent_roles.get(agent_idx, f"Agent {agent_idx+1} analyzing and responding to previous content.")
                
                # Build the system message with enhanced collaborative instructions
                system_message = (
                    f"You are Agent {agent_idx+1}. {agent_role}\n\n"
                    f"You are participating in a multi-agent discussion with {self.num_agents-1} other AI agents.\n"
                    f"Each agent has different expertise and perspective. Your goal is to:\n"
                    f"1. Address the user's query based on your specific role and expertise\n"
                    f"2. Respond to and build upon the contributions of previous agents\n"
                    f"3. Add unique value through your specific perspective\n"
                    f"4. Identify any gaps or disagreements with previous responses\n"
                    f"5. Move the conversation forward with depth and insight"
                )
                
                # If not the first agent, explicitly instruct to respond to previous agents
                if len(self.conversation_history) > 1 and any(("Agent " in entry.get("role", "")) for entry in self.conversation_history):
                    system_message += "\n\nImportant: Directly engage with and reference what previous agents have said. Don't just provide your own standalone response."
                
                # Build messages array for OpenAI
                messages = [
                    {"role": "system", "content": system_message},
                ]
                
                # Add user query with additional context based on position in conversation
                if agent_idx == 0:  # First agent
                    messages.append({"role": "user", "content": self.prompt})
                else:  # Subsequent agents - add context about the discussion
                    context = f"This is a multi-agent discussion. The original user question was: {self.prompt}\n\nPlease review what the previous agents have said and contribute your perspective."
                    messages.append({"role": "user", "content": context})
                
                # Add previous agent responses - formated to encourage interaction
                if len(self.conversation_history) > 1:
                    # For continuous mode, limit history to avoid token limits
                    max_history = 10 if self.continuous_mode else len(self.conversation_history) - 1
                    history_to_include = self.conversation_history[-max_history:] if len(self.conversation_history) - 1 > max_history else self.conversation_history[:-1]
                    
                    for entry in history_to_include:
                        if entry["role"] == "user":
                            messages.append({"role": "user", "content": f"User asked: {entry['content']}"})
                        else:  # This is an agent response
                            # Format agent responses to encourage interaction
                            agent_id = entry["role"]
                            content = entry["content"]
                            messages.append({"role": "assistant", "content": f"{agent_id}: {content}"})
                    
                    # Add specific instruction to respond to previous agents
                    if agent_idx > 0 or self.current_turn > 0:
                        latest_agent_responses = [entry for entry in self.conversation_history[-3:] if entry["role"].startswith("Agent")]
                        if latest_agent_responses:
                            agent_references = ", ".join([entry["role"] for entry in latest_agent_responses])
                            messages.append({
                                "role": "user", 
                                "content": f"Now it's your turn as Agent {agent_idx+1}. Please consider and respond to what {agent_references} said above."
                            })
                
                # Configure generation settings
                generation_config = {
                    "model": self.model_name,
                    "temperature": 0.8,  # Slightly higher temperature for more varied responses
                    "top_p": 1.0,
                    "max_tokens": 1024,  # Limit token length for faster responses in continuous mode
                    "messages": messages
                }
                
                # Generate the agent's response
                response = client.chat.completions.create(**generation_config)
                
                # Extract the response text
                if response and response.choices and len(response.choices) > 0:
                    agent_response = response.choices[0].message.content
                else:
                    agent_response = f"Agent {agent_idx+1} could not generate a response."
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": f"Agent {agent_idx+1}",
                    "content": agent_response
                })
                
                # Emit the response signal
                self.agent_response.emit(agent_idx, agent_response)
                
                # If not in continuous mode, or stop requested, break after all agents have responded once
                if not self.continuous_mode:
                    if agent_idx == self.num_agents - 1:
                        break
                
                # Move to next agent in rotation
                agent_idx = (agent_idx + 1) % self.num_agents
                
                # Increment turn counter when we've gone through all agents
                if agent_idx == 0:
                    self.current_turn += 1
                
                # Small delay to allow UI to update
                time.sleep(0.5)
            
            # Signal completion of the multi-agent dialog
            self.dialog_complete.emit()
            
        except Exception as e:
            logging.error(f"Failed to generate multi-agent dialog: {e}")
            self.dialog_error.emit(str(e))

# ------------------------------------------------------------------------------
# Main Entry Point
def check_dependencies():
    """Check that all required dependencies are installed"""
    required_packages = []
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("Please install them using: pip install " + " ".join(missing))
        return False
    return True

def setup_exception_handling():
    """Set up global exception handler to log unhandled exceptions"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        
    sys.excepthook = handle_exception

if __name__ == "__main__":
    try:
        # Set up exception handling first
        setup_exception_handling()
        
        # Check dependencies before starting
        if not check_dependencies():
            sys.exit(1)
            
        # Initialize the application with error handling
        app = QApplication(sys.argv)
        
        # Load environment variables if .env file exists
        if os.path.exists(".env"):
            load_dotenv()
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            QMessageBox.warning(None, "API Key Missing", 
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to a .env file.")
        
        # Create main window with exception handling
        window = GeminiChatApp()
        window.setWindowTitle("OpenAI Chat Enhanced")
        
        # Show the window
        window.show()
        
        # Start the event loop
        sys.exit(app.exec_())
    except Exception as e:
        # Show error in message box for better debugging
        if QApplication.instance():
            QMessageBox.critical(None, "Startup Error", f"Application failed to start: {str(e)}")
        logging.critical(f"Application startup failed: {e}", exc_info=True)
        sys.exit(1)


