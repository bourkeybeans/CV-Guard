"""LinkdAPI client wrapper"""

import asyncio
from enum import Enum
from typing import Optional, Dict, Set, Tuple, Union
from linkdapi import AsyncLinkdAPI
from rich.console import Console

console = Console()


class ScrapeStatus(Enum):
    """Status codes for scraping operations"""
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


class LinkedInAPIClient:
    """Wrapper for LinkdAPI client"""

    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: int = 5, log_callback=None):
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_callback = log_callback

    def _log(self, message: str):
        """Log a message using callback if available, otherwise print to console"""
        if self.log_callback:
            self.log_callback(message)
        else:
            console.print(message)

    def _check_response_success(self, response: Dict, endpoint_name: str) -> Tuple[bool, Optional[str], bool]:
        """
        Check if API response is successful
        Returns: (is_success, error_message, is_retryable)
        """
        if not response:
            return False, "Empty response", True

        if 'error' in response:
            error = response['error']
            if isinstance(error, dict):
                message = error.get('message', str(error))
                return False, f"API Error: {message}", True
            return False, f"API Error: {error}", True

        if 'success' in response and not response['success']:
            message = response.get('message', 'Unknown error')
            non_retryable_keywords = [
                "cannot be displayed", "doesn't exist", "not found", 
                "invalid username", "profile not available", "does not exist"
            ]
            is_retryable = not any(keyword in message.lower() for keyword in non_retryable_keywords)
            return False, f"Request failed: {message}", is_retryable

        if 'status' in response:
            status = response['status']
            if status == 429:
                return False, "RATE_LIMIT", True
            elif status == 404:
                return False, "Profile not found", False
            elif status >= 400:
                message = response.get('message', f'HTTP {status}')
                return False, f"HTTP Error {status}: {message}", True

        if 'data' not in response and endpoint_name != 'overview':
            return False, "No data in response", True

        return True, None, True

    async def _make_api_call_with_retry(self, api_call, endpoint_name: str, identifier: str):
        """
        Make an API call with retry logic for rate limits and failures
        Returns: (status, response_or_error)
        """
        last_was_rate_limit = False
        for attempt in range(self.max_retries):
            try:
                response = await api_call()
                is_success, error_msg, is_retryable = self._check_response_success(response, endpoint_name)

                if is_success:
                    return ScrapeStatus.SUCCESS, response

                # Check if this is a "not found" type error
                if not is_retryable:
                    return ScrapeStatus.NOT_FOUND, error_msg

                # Handle rate limits
                if error_msg == "RATE_LIMIT":
                    last_was_rate_limit = True
                    wait_time = self.retry_delay * (attempt + 1)
                    if attempt == 0:
                        self._log(f"[yellow]⏳ ({identifier}) Rate limit hit - waiting {wait_time}s[/]")
                    await asyncio.sleep(wait_time)
                    continue

                # Handle other retryable errors
                if attempt < self.max_retries - 1:
                    wait_time = 1
                    if attempt == 0:
                        clean_msg = error_msg.replace("Request failed: ", "").replace("API Error: ", "")
                        self._log(f"[yellow]⚠ ({identifier}) {clean_msg[:60]}... retrying in {wait_time}s[/]")
                    await asyncio.sleep(wait_time)
                else:
                    return ScrapeStatus.ERROR, error_msg

            except Exception as e:
                error_msg = str(e)
                # Check if this is a rate limit error (case insensitive)
                if "429" in error_msg or "too many requests" in error_msg.lower():
                    last_was_rate_limit = True
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (attempt + 1)
                        if attempt == 0:
                            self._log(f"[yellow]⏳ ({identifier}) Rate limit hit - waiting {wait_time}s[/]")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Exhausted retries on rate limit
                        return ScrapeStatus.RATE_LIMITED, "Rate limit exceeded after all retries"

                if attempt < self.max_retries - 1:
                    wait_time = 1
                    if attempt == 0:
                        # Clean up error message for display
                        clean_msg = error_msg.replace("Client error", "").replace("for url", "").strip()
                        self._log(f"[yellow]⚠ ({identifier}) {clean_msg[:60]}... retrying in {wait_time}s[/]")
                    await asyncio.sleep(wait_time)
                else:
                    return ScrapeStatus.ERROR, error_msg

        # Exhausted retries - determine if it was rate limit or other error
        if last_was_rate_limit:
            return ScrapeStatus.RATE_LIMITED, "Rate limit exceeded after all retries"
        return ScrapeStatus.ERROR, "Max retries exceeded"

    async def fetch_profile_data(
        self,
        api: AsyncLinkdAPI,
        username: str,
        data_options: Set[str] = None
    ) -> Tuple[ScrapeStatus, Union[Dict, str]]:
        """
        Fetch profile data for a single username based on selected options
        Returns: (status, profile_data_or_error_message)
        """
        if data_options is None:
            data_options = {'overview', 'details', 'experience', 'education', 'skills', 'certifications', 'contact'}

        try:
            status, overview_or_error = await self._make_api_call_with_retry(
                lambda: api.get_profile_overview(username),
                'overview',
                username
            )
            if status != ScrapeStatus.SUCCESS:
                return status, overview_or_error

            if 'data' not in overview_or_error:
                return ScrapeStatus.ERROR, "No data in overview response"

            profile_data = overview_or_error['data']
            urn = profile_data.get('urn', '')

            if not urn:
                profile_data['username'] = username
                return ScrapeStatus.SUCCESS, profile_data

            # Create tasks for all additional data calls
            tasks = []

            if 'details' in data_options:
                tasks.append(self._fetch_and_process(api, username, urn, 'details', 'get_profile_details'))
            if 'experience' in data_options:
                tasks.append(self._fetch_and_process(api, username, urn, 'experience', 'get_full_experience'))
            if 'education' in data_options:
                tasks.append(self._fetch_and_process(api, username, urn, 'education', 'get_education'))
            if 'skills' in data_options:
                tasks.append(self._fetch_and_process(api, username, urn, 'skills', 'get_skills'))
            if 'certifications' in data_options:
                tasks.append(self._fetch_and_process(api, username, urn, 'certifications', 'get_certifications'))
            if 'contact' in data_options:
                tasks.append(self._fetch_and_process(api, username, username, 'contact', 'get_contact_info'))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if result and not isinstance(result, Exception):
                        self._update_profile_data(profile_data, result)

            profile_data['username'] = username
            return ScrapeStatus.SUCCESS, profile_data

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)[:60]}"
            self._log(f"[red]✗ ({username}) {error_msg}...[/]")
            return ScrapeStatus.ERROR, error_msg

    async def _fetch_and_process(self, api: AsyncLinkdAPI, username: str, identifier: str, data_type: str, method_name: str):
        """Fetch and process a specific data type"""
        try:
            method = getattr(api, method_name)
            status, result = await self._make_api_call_with_retry(
                lambda: method(identifier),
                data_type,
                username
            )

            # Check if the API call was successful
            if status != ScrapeStatus.SUCCESS:
                # Only log if it's a real error, not just "not found" (which is expected for some profiles)
                if status == ScrapeStatus.NOT_FOUND:
                    self._log(f"[dim]⚠ ({username}) no {data_type} available[/]")
                else:
                    self._log(f"[yellow]⚠ ({username}) {data_type} failed: {result[:60] if isinstance(result, str) else 'Unknown error'}[/]")
                return None

            # Check if we have a valid response with data
            if not result or not isinstance(result, dict):
                self._log(f"[yellow]⚠ ({username}) invalid response for {data_type}[/]")
                return None

            # Check if data exists in response
            if 'data' not in result:
                self._log(f"[dim]⚠ ({username}) no {data_type} data in response[/]")
                return None

            data = result['data']
            
            # Check if data is actually empty (empty list, empty dict, None, etc.)
            if data is None:
                self._log(f"[dim]⚠ ({username}) no {data_type} available[/]")
                return None
            
            # For list/dict data, check if it's empty
            if isinstance(data, (list, dict)) and len(data) == 0:
                self._log(f"[dim]⚠ ({username}) no {data_type} available[/]")
                return None

            # Data exists and is not empty
            return {
                'type': data_type,
                'data': data
            }

        except Exception as e:
            self._log(f"[yellow]⚠ ({username}) {data_type} failed: {str(e)[:60]}...[/]")
            return None

    def _update_profile_data(self, profile_data: Dict, result: Dict):
        """Update profile data with the result from an API call"""
        data_type = result['type']
        data = result['data']

        if data_type == 'details':
            profile_data['about'] = data.get('about', '')
            profile_data['positions'] = data.get('positions', [])
            profile_data['education'] = data.get('education', [])
        elif data_type == 'experience':
            profile_data['experience'] = data.get('experience', []) if isinstance(data, dict) else data
        elif data_type == 'education':
            profile_data['education'] = data.get('education', []) if isinstance(data, dict) else data
        elif data_type == 'skills':
            profile_data['skills'] = data.get('skills', []) if isinstance(data, dict) else data
        elif data_type == 'certifications':
            profile_data['certifications'] = data.get('certifications', []) if isinstance(data, dict) else data
        elif data_type == 'contact':
            profile_data['contactInfo'] = data
