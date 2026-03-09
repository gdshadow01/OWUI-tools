from requests import post
from pathlib import Path

def upload_file_to_owui(url: str, token: str, file_path: str, filename: str = None) -> dict:
    """
    Upload a file from workspace to OpenWebUI API.

    Args:
        url: OpenWebUI base URL (e.g., "http://host.docker.internal:3000")
        token: Authorization header (full "Bearer ..." string)
        file_path: Full path to file in workspace
        filename: Desired filename in OpenWebUI (defaults to original filename)

    Returns:
        dict: {"file_path_download": "[Download link]"} or {"error": {...}}
    """
    mime_types = {
        'txt': 'text/plain',
        'md': 'text/markdown',
        'py': 'text/plain',
        'js': 'text/javascript',
        'json': 'application/json',
        'csv': 'text/csv',
        'yaml': 'text/yaml',
        'yml': 'text/yaml',
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'zip': 'application/zip'
    }

    file_obj = Path(file_path)
    if filename is None:
        filename = file_obj.name

    ext = filename.split('.')[-1].lower() if '.' in filename else 'txt'
    mime_type = mime_types.get(ext, 'application/octet-stream')

    with open(file_path, 'rb') as f:
        files = {'file': (filename, f, mime_type)}
        headers = {
            'Authorization': token,
            'Accept': 'application/json'
        }
        params = {"process": "true", "process_in_background": "false"}

        response = post(
            f'{url}/api/v1/files/',
            headers=headers,
            files=files,
            params=params,
            timeout=60
        )

        if response.status_code != 200:
            return {
                "error": {
                    "message": f'Upload failed: {response.status_code} - {response.text}'
                }
            }

        data = response.json()
        file_id = data.get('id')

        return {
            "file_path_download": f"[Download {filename}](/api/v1/files/{file_id}/content)",
            "file_id": file_id,
            "filename": filename
        }
