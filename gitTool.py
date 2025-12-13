#!/usr/bin/env python3
"""
GitHub Repository Fetcher
Fetches all repositories for a given GitHub account and returns detailed information in JSON format.
"""

import requests
import json
import sys
import argparse
from typing import List, Dict, Optional


def get_github_repositories(username: str, token: Optional[str] = None) -> List[Dict]:
    """
    Fetch all repositories for a given GitHub username.
    
    Args:
        username: GitHub username or organization name
        token: Optional GitHub personal access token for higher rate limits
        
    Returns:
        List of dictionaries containing repository details
    """
    base_url = f"https://api.github.com/users/{username}/repos"
    headers = {}
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    repositories = []
    page = 1
    per_page = 100  # Maximum allowed by GitHub API
    
    while True:
        params = {
            "page": page,
            "per_page": per_page,
            "type": "all"  # Get all repos (public, private if authenticated)
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            
            repos = response.json()
            
            # If no more repositories, break
            if not repos:
                break
            
            # Process each repository
            for repo in repos:
                repo_details = {
                    "id": repo.get("id"),
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "description": repo.get("description"),
                    "url": repo.get("html_url"),
                    "clone_url": repo.get("clone_url"),
                    "ssh_url": repo.get("ssh_url"),
                    "git_url": repo.get("git_url"),
                    "language": repo.get("language"),
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "watchers": repo.get("watchers_count", 0),
                    "open_issues": repo.get("open_issues_count", 0),
                    "is_private": repo.get("private", False),
                    "is_fork": repo.get("fork", False),
                    "is_archived": repo.get("archived", False),
                    "is_disabled": repo.get("disabled", False),
                    "created_at": repo.get("created_at"),
                    "updated_at": repo.get("updated_at"),
                    "pushed_at": repo.get("pushed_at"),
                    "default_branch": repo.get("default_branch"),
                    "size": repo.get("size", 0),  # Size in KB
                    "topics": repo.get("topics", []),
                    "license": repo.get("license", {}).get("name") if repo.get("license") else None,
                    "has_issues": repo.get("has_issues", False),
                    "has_projects": repo.get("has_projects", False),
                    "has_wiki": repo.get("has_wiki", False),
                    "has_pages": repo.get("has_pages", False),
                    "has_downloads": repo.get("has_downloads", False),
                    "homepage": repo.get("homepage"),
                }
                repositories.append(repo_details)
            
            # Check if there are more pages
            # GitHub API uses Link header for pagination
            link_header = response.headers.get("Link", "")
            if "rel=\"next\"" not in link_header:
                break
            
            page += 1
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"Error: User or organization '{username}' not found.", file=sys.stderr)
                sys.exit(1)
            elif response.status_code == 403:
                print(f"Error: Rate limit exceeded. Consider using a GitHub token.", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to GitHub API: {e}", file=sys.stderr)
            sys.exit(1)
    
    return repositories


def main():
    parser = argparse.ArgumentParser(
        description="Fetch all repositories for a GitHub account and return JSON"
    )
    parser.add_argument(
        "username",
        help="GitHub username or organization name"
    )
    parser.add_argument(
        "-t", "--token",
        help="GitHub personal access token (optional, increases rate limit)",
        default=None
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: {username}_repositories.json)",
        default=None
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of saving to file"
    )
    
    args = parser.parse_args()
    
    # Fetch repositories
    repos = get_github_repositories(args.username, args.token)
    
    # Prepare output
    output_data = {
        "username": args.username,
        "total_repositories": len(repos),
        "repositories": repos
    }
    
    # Format JSON
    if args.pretty:
        json_output = json.dumps(output_data, indent=2, ensure_ascii=False)
    else:
        json_output = json.dumps(output_data, ensure_ascii=False)
    
    # Write output
    if args.stdout:
        # Print to stdout
        print(json_output)
    else:
        # Save to file
        output_file = args.output or f"{args.username}_repositories.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Successfully fetched {len(repos)} repositories. Output saved to {output_file}")


if __name__ == "__main__":
    main()

