from requests import get
from io import BytesIO

def download_file_from_owui(url: str, token: str, file_id: str) -> BytesIO | dict:
    """
    Download a file from OpenWebUI API.

    Args:
        url: OpenWebUI base URL
        token: Authorization header (full "Bearer ..." string)
        file_id: ID of file to download

    Returns:
        BytesIO: File content in memory buffer (success)
        dict: {"error": {"message": "..."}} (failure)
    """
    download_url = f'{url}/api/v1/files/{file_id}/content'

    headers = {
        'Authorization': token,
        'Accept': 'application/json'
    }

    try:
        response = get(download_url, headers=headers, timeout=60)

        if response.status_code != 200:
            return {
                "error": {
                    "message": f'Download failed: {response.status_code} - {response.text}'
                }
            }

        return BytesIO(response._content)
    
    except Exception as e:
        return {
            "error": {
                "message": f'Download failed with exception: {str(e)}'
            }
        }
