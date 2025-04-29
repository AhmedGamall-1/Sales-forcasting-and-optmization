import requests
from typing import Dict, Any
from ..config import settings

class APIService:
    def __init__(self):
        self.base_url = settings.API_BASE_URL
    
    def predict_sales(self, car_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction request to the API."""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=car_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {str(e)}")

api_service = APIService() 