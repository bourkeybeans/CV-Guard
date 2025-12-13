"""OpenAI Agent for LinkedIn Profile Scraping"""

import asyncio
import json
from typing import Dict, List, Optional, Set, Any
from openai import OpenAI
from rich.console import Console

from ..scraper.scraper import ProfileScraper
from ..api.client import ScrapeStatus

console = Console()


class LinkedInScrapingAgent:
    """Agent that uses OpenAI function calling to scrape LinkedIn profiles"""

    def __init__(
        self,
        openai_api_key: str,
        linkdapi_key: str,
        model: str = "gpt-4o",
        max_concurrent: int = 10,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.scraper = ProfileScraper(
            api_key=linkdapi_key,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        self.tools = self._define_tools()
        self.conversation_history: List[Dict[str, Any]] = []

    def _define_tools(self) -> List[Dict]:
        """Define the tools/functions available to the OpenAI agent"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "scrape_linkedin_profile",
                    "description": "Scrape a LinkedIn profile by username. Returns comprehensive profile data including overview, details, experience, education, skills, certifications, and contact information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "The LinkedIn username (public identifier) to scrape. For example, 'john-doe' for linkedin.com/in/john-doe"
                            },
                            "data_options": {
                                "type": "object",
                                "description": "Which data sections to scrape. Overview is always included.",
                                "properties": {
                                    "details": {
                                        "type": "boolean",
                                        "description": "Include detailed about section and positions",
                                        "default": True
                                    },
                                    "experience": {
                                        "type": "boolean",
                                        "description": "Include work experience history",
                                        "default": True
                                    },
                                    "education": {
                                        "type": "boolean",
                                        "description": "Include education history",
                                        "default": True
                                    },
                                    "skills": {
                                        "type": "boolean",
                                        "description": "Include skills and endorsements",
                                        "default": True
                                    },
                                    "certifications": {
                                        "type": "boolean",
                                        "description": "Include certifications",
                                        "default": True
                                    },
                                    "contact": {
                                        "type": "boolean",
                                        "description": "Include contact information (email, phone, websites)",
                                        "default": True
                                    }
                                }
                            }
                        },
                        "required": ["username"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "scrape_multiple_profiles",
                    "description": "Scrape multiple LinkedIn profiles at once. More efficient for batch operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "usernames": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of LinkedIn usernames to scrape"
                            },
                            "data_options": {
                                "type": "object",
                                "description": "Which data sections to scrape. Overview is always included.",
                                "properties": {
                                    "details": {
                                        "type": "boolean",
                                        "description": "Include detailed about section and positions",
                                        "default": True
                                    },
                                    "experience": {
                                        "type": "boolean",
                                        "description": "Include work experience history",
                                        "default": True
                                    },
                                    "education": {
                                        "type": "boolean",
                                        "description": "Include education history",
                                        "default": True
                                    },
                                    "skills": {
                                        "type": "boolean",
                                        "description": "Include skills and endorsements",
                                        "default": True
                                    },
                                    "certifications": {
                                        "type": "boolean",
                                        "description": "Include certifications",
                                        "default": True
                                    },
                                    "contact": {
                                        "type": "boolean",
                                        "description": "Include contact information (email, phone, websites)",
                                        "default": True
                                    }
                                }
                            }
                        },
                        "required": ["usernames"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_scraped_profiles",
                    "description": "Get all previously scraped profiles in the current session",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

    async def _execute_scrape_profile(
        self,
        username: str,
        data_options: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """Execute the scrape_profile function"""
        try:
            # Convert data_options dict to set format expected by scraper
            options_set = {'overview'}  # Always include overview
            if data_options:
                if data_options.get('details', True):
                    options_set.add('details')
                if data_options.get('experience', True):
                    options_set.add('experience')
                if data_options.get('education', True):
                    options_set.add('education')
                if data_options.get('skills', True):
                    options_set.add('skills')
                if data_options.get('certifications', True):
                    options_set.add('certifications')
                if data_options.get('contact', True):
                    options_set.add('contact')
            else:
                # Default: include everything
                options_set.update(['details', 'experience', 'education', 'skills', 'certifications', 'contact'])

            # Scrape the profile
            await self.scraper.scrape_profiles([username], options_set)
            
            # Get the scraped data
            profiles = self.scraper.get_profiles_data()
            
            # Find the profile we just scraped
            for profile in profiles:
                if profile.get('publicIdentifier', '').lower() == username.lower() or \
                   profile.get('username', '').lower() == username.lower():
                    return {
                        "success": True,
                        "profile": profile,
                        "message": f"Successfully scraped profile for {username}"
                    }
            
            # Check if it failed
            failed = self.scraper.get_failed_profiles()
            for failed_profile in failed:
                if failed_profile.get('username', '').lower() == username.lower():
                    return {
                        "success": False,
                        "error": failed_profile.get('error', 'Unknown error'),
                        "message": f"Failed to scrape profile for {username}: {failed_profile.get('error', 'Unknown error')}"
                    }
            
            return {
                "success": False,
                "error": "Profile not found in results",
                "message": f"Could not find scraped data for {username}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error scraping profile: {str(e)}"
            }

    async def _execute_scrape_multiple_profiles(
        self,
        usernames: List[str],
        data_options: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """Execute the scrape_multiple_profiles function"""
        try:
            # Convert data_options dict to set format expected by scraper
            options_set = {'overview'}  # Always include overview
            if data_options:
                if data_options.get('details', True):
                    options_set.add('details')
                if data_options.get('experience', True):
                    options_set.add('experience')
                if data_options.get('education', True):
                    options_set.add('education')
                if data_options.get('skills', True):
                    options_set.add('skills')
                if data_options.get('certifications', True):
                    options_set.add('certifications')
                if data_options.get('contact', True):
                    options_set.add('contact')
            else:
                # Default: include everything
                options_set.update(['details', 'experience', 'education', 'skills', 'certifications', 'contact'])

            # Scrape the profiles
            await self.scraper.scrape_profiles(usernames, options_set)
            
            # Get the scraped data
            profiles = self.scraper.get_profiles_data()
            failed = self.scraper.get_failed_profiles()
            
            # Match profiles to usernames
            results = []
            for username in usernames:
                found = False
                for profile in profiles:
                    if profile.get('publicIdentifier', '').lower() == username.lower() or \
                       profile.get('username', '').lower() == username.lower():
                        results.append({
                            "username": username,
                            "success": True,
                            "profile": profile
                        })
                        found = True
                        break
                
                if not found:
                    # Check failed profiles
                    error_msg = "Not found"
                    for failed_profile in failed:
                        if failed_profile.get('username', '').lower() == username.lower():
                            error_msg = failed_profile.get('error', 'Unknown error')
                            break
                    
                    results.append({
                        "username": username,
                        "success": False,
                        "error": error_msg
                    })
            
            return {
                "success": True,
                "results": results,
                "message": f"Scraped {len([r for r in results if r['success']])} out of {len(usernames)} profiles"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error scraping profiles: {str(e)}"
            }

    def _execute_get_scraped_profiles(self) -> Dict[str, Any]:
        """Execute the get_scraped_profiles function"""
        profiles = self.scraper.get_profiles_data()
        return {
            "success": True,
            "count": len(profiles),
            "profiles": profiles,
            "message": f"Found {len(profiles)} scraped profile(s)"
        }

    async def _execute_function(self, function_name: str, arguments: Dict) -> Dict[str, Any]:
        """Execute a function call from the agent"""
        if function_name == "scrape_linkedin_profile":
            username = arguments.get("username")
            data_options = arguments.get("data_options")
            return await self._execute_scrape_profile(username, data_options)
        
        elif function_name == "scrape_multiple_profiles":
            usernames = arguments.get("usernames", [])
            data_options = arguments.get("data_options")
            return await self._execute_scrape_multiple_profiles(usernames, data_options)
        
        elif function_name == "get_scraped_profiles":
            return self._execute_get_scraped_profiles()
        
        else:
            return {
                "success": False,
                "error": f"Unknown function: {function_name}",
                "message": f"Function {function_name} is not available"
            }

    async def chat(self, user_message: str, system_message: Optional[str] = None) -> str:
        """
        Chat with the agent to scrape LinkedIn profiles
        
        Args:
            user_message: The user's request/question
            system_message: Optional system message to set agent behavior
        
        Returns:
            The agent's response
        """
        if system_message is None:
            system_message = """You are a helpful assistant that can scrape LinkedIn profiles using the provided tools.
When a user asks you to scrape a profile, use the scrape_linkedin_profile function.
You can scrape multiple profiles at once using scrape_multiple_profiles.
Always provide clear, helpful responses about what data was scraped and any issues encountered.
If a profile is not found or fails to scrape, explain the error clearly."""

        # Add user message to conversation
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Make the API call
        messages = [
            {"role": "system", "content": system_message}
        ] + self.conversation_history

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            
            # Add assistant message to conversation
            self.conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in (message.tool_calls or [])
                ]
            })

            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    # Execute the function
                    function_result = await self._execute_function(function_name, arguments)

                    # Add tool result to conversation
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(function_result, default=str)
                    })

                # Get the final response after tool execution
                messages = [
                    {"role": "system", "content": system_message}
                ] + self.conversation_history

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )

                final_message = response.choices[0].message
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_message.content
                })

                return final_message.content or "Task completed."
            else:
                return message.content or "I'm ready to help you scrape LinkedIn profiles."

        except Exception as e:
            error_msg = f"Error in agent chat: {str(e)}"
            console.print(f"[red]{error_msg}[/]")
            return error_msg

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

