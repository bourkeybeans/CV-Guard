"""Configuration management module"""

import configparser
import os
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()


class Config:
    """Configuration manager for the LinkedIn scraper"""

    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> configparser.ConfigParser:
        """Load configuration from config.ini"""
        config = configparser.ConfigParser()

        if not os.path.exists(self.config_path):
            console.print(f"[bold red]Error:[/] Config file '{self.config_path}' not found!")
            console.print("[yellow]Please create config.ini with your LinkdAPI key[/]")
            sys.exit(1)

        config.read(self.config_path)

        # Validate API key
        api_key = config.get('LINKDAPI', 'api_key', fallback='YOUR_API_KEY_HERE')
        if api_key == 'YOUR_API_KEY_HERE' or not api_key:
            console.print("[bold red]Error:[/] Please set your LinkdAPI key in config.ini")
            console.print("[cyan]Get your API key from: https://linkdapi.com[/]")
            sys.exit(1)

        return config

    @property
    def api_key(self) -> str:
        """Get the API key"""
        return self.config.get('LINKDAPI', 'api_key')

    @property
    def max_concurrent(self) -> int:
        """Get the maximum concurrent requests"""
        return int(self.config.get('SETTINGS', 'max_concurrent_requests', fallback='10'))

    @property
    def output_dir(self) -> str:
        """Get the output directory"""
        output_dir = self.config.get('SETTINGS', 'output_directory', fallback='output')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    @property
    def max_retries(self) -> int:
        """Get the maximum number of retries"""
        return int(self.config.get('SETTINGS', 'max_retries', fallback='3'))

    @property
    def retry_delay(self) -> int:
        """Get the retry delay in seconds"""
        return int(self.config.get('SETTINGS', 'retry_delay', fallback='1'))

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get the OpenAI API key (optional)"""
        try:
            api_key = self.config.get('OPENAI', 'api_key', fallback='')
            if api_key and api_key != 'YOUR_OPENAI_API_KEY_HERE':
                return api_key
        except:
            pass
        return None

    @property
    def openai_model(self) -> str:
        """Get the OpenAI model to use"""
        try:
            return self.config.get('OPENAI', 'model', fallback='gpt-4o')
        except:
            return 'gpt-4o'

