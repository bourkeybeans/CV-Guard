#!/usr/bin/env python3
"""LinkedIn Profile Scraper Agent - OpenAI Function Calling"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

sys.path.insert(0, str(Path(__file__).parent))

from src.agent.agent import LinkedInScrapingAgent
from src.utils.config import Config

console = Console()


def show_header():
    """Display the agent header"""
    header = """
[bold cyan]╔═══════════════════════════════════════════════════════════╗[/]
[bold cyan]║[/]  [bold white]LinkedIn Profile Scraper Agent (OpenAI Powered)[/]  [bold cyan]║[/]
[bold cyan]╚═══════════════════════════════════════════════════════════╝[/]
"""
    console.print(header)


async def main():
    """Main entry point for the agent"""
    show_header()

    # Load configuration
    try:
        config = Config()
    except SystemExit:
        console.print("[red]Configuration error. Please check config.ini[/]")
        return

    # Check for OpenAI API key
    openai_api_key = config.openai_api_key
    if not openai_api_key:
        console.print("[yellow]OpenAI API key not found in config.ini[/]")
        console.print("[cyan]Please add your OpenAI API key to config.ini:[/]")
        console.print("[dim]  [OPENAI][/]")
        console.print("[dim]  api_key = your-openai-api-key-here[/]")
        console.print("[dim]  model = gpt-4o[/]")
        
        # Allow manual entry
        manual_key = Prompt.ask("\n[cyan]Enter OpenAI API key[/] (or press Enter to exit)", password=True)
        if not manual_key:
            console.print("[yellow]Exiting...[/]")
            return
        openai_api_key = manual_key
    else:
        console.print(f"[green]✓[/] OpenAI API key loaded")
        console.print(f"[green]✓[/] Using model: {config.openai_model}")

    # Initialize agent
    console.print(f"[green]✓[/] LinkdAPI key loaded")
    console.print(f"[green]✓[/] Initializing agent...\n")

    agent = LinkedInScrapingAgent(
        openai_api_key=openai_api_key,
        linkdapi_key=config.api_key,
        model=config.openai_model,
        max_concurrent=config.max_concurrent,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay
    )

    console.print(Panel(
        "[bold cyan]Agent ready![/]\n\n"
        "You can now ask the agent to scrape LinkedIn profiles.\n"
        "Examples:\n"
        "  • 'Scrape the profile for username john-doe'\n"
        "  • 'Get profile data for farhan'\n"
        "  • 'Scrape profiles for john-doe, jane-smith, and bob-jones'\n"
        "  • 'Show me all scraped profiles'\n\n"
        "Type 'exit' or 'quit' to end the session.",
        title="[bold]Welcome[/]",
        border_style="cyan"
    ))

    # Interactive loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/]")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("\n[cyan]Goodbye![/]")
                break

            if not user_input.strip():
                continue

            # Show thinking indicator
            with console.status("[bold cyan]Agent is thinking...", spinner="dots"):
                response = await agent.chat(user_input)

            console.print(f"\n[bold green]Agent:[/] {response}")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted by user[/]")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/] {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[cyan]Goodbye![/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Fatal error:[/] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

