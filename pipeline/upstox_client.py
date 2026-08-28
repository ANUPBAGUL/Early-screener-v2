import requests
import urllib.parse
import os
import json
from datetime import datetime
from dotenv import load_dotenv

class UpstoxClient:
    def __init__(self, api_key=None, api_secret=None, redirect_uri=None):
        # Load from environment or args
        load_dotenv(override=True)
        self.api_key = api_key or os.getenv("UPSTOX_API_KEY")
        self.api_secret = api_secret or os.getenv("UPSTOX_API_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("UPSTOX_REDIRECT_URI")
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self.base_url = "https://api.upstox.com/v2"
        
    def get_login_url(self):
        """Generates the URL the user needs to visit to authorize the app."""
        params = {
            "response_type": "code",
            "client_id": self.api_key,
            "redirect_uri": self.redirect_uri
        }
        query_string = urllib.parse.urlencode(params)
        return f"{self.base_url}/login/authorization/dialog?{query_string}"
        
    def get_access_token(self, auth_code):
        """Exchanges the auth code for an access token."""
        url = f"{self.base_url}/login/authorization/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "code": auth_code,
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code"
        }
        
        response = requests.post(url, headers=headers, data=data, timeout=10)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data.get("access_token")
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.text}")
            
    def _get_headers(self):
        if not self.access_token:
            raise ValueError("Access token is missing. Please authenticate first.")
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

    def fetch_historical_candles(self, instrument_key, interval="day", to_date=None, from_date=None):
        """
        Fetches historical OHLCV data.
        interval: '1minute', 'day', 'week', 'month'
        Dates should be in 'YYYY-MM-DD' format.
        """
        if to_date is None:
            to_date = datetime.today().strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}"
        if from_date:
            url += f"/{from_date}"
            
        response = requests.get(url, headers=self._get_headers(), timeout=10)
        if response.status_code == 200:
            return response.json().get("data", {}).get("candles", [])
        else:
            raise Exception(f"Failed to fetch candles for {instrument_key}: {response.text}")

# Helper for testing auth flow
if __name__ == "__main__":
    print("--- Upstox API Authentication ---")
    client = UpstoxClient()
    if not client.api_key or not client.api_secret:
        print("Error: API credentials missing. Check your .env file.")
    else:
        print("\n1. Visit this URL to authorize the app:")
        print(client.get_login_url())
        
        print("\n2. After logging in, you will be redirected to your Redirect URI.")
        print("   The URL will look like: https://localhost:8501/?code=XXXXXX")
        auth_code = input("\n3. Paste the 'code' parameter from the URL here: ").strip()
        
        if auth_code:
            try:
                token = client.get_access_token(auth_code)
                print(f"\nSuccess! Access token obtained.")
                
                # Append to .env file
                env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
                with open(env_path, "a") as f:
                    f.write(f'\nUPSTOX_ACCESS_TOKEN="{token}"\n')
                print(f"Token saved to {env_path}. You can now run build_db.py!")
            except Exception as e:
                print(f"Error getting token: {e}")

