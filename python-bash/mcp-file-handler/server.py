from os import getenv
from typing import Annotated
from json import dumps
import logging

from pydantic import Field, BaseModel
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
import requests

from utils import (
    upload_file_to_owui,
    download_file_from_owui,
    save_to_workspace,
    read_from_workspace,
    list_workspace_files,
    delete_workspace_file,
    move_to_outputs
)

OWUI_URL = getenv('OWUI_URL', 'http://host.docker.internal:3000')
PORT = int(getenv('PORT', 8015))
WORKSPACE_DIR = getenv('WORKSPACE_DIR', '/workspace')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


mcp = FastMCP(
    name="file-handler",
    instructions="Handles file uploads, downloads, and workspace management for OpenWebUI.",
    port=PORT,
    host="0.0.0.0"
)


class FileUploadRequest(BaseModel):
    file_id: str
    filename: str = None


@mcp.tool(
    name="upload_file_from_openwebui",
    title="Upload file from OpenWebUI to workspace",
    description="""Download a file uploaded to OpenWebUI and save it to the workspace.

    Use this when a user has uploaded a file through OpenWebUI's UI and you need to access it in Python code.

    Args:
        file_id: The ID of the file in OpenWebUI (available in file metadata)
        filename: Optional custom filename. If not provided, uses the original filename.
    """
)
async def upload_file_from_openwebui(
    file_id: Annotated[str, Field(description="The ID of the file in OpenWebUI")],
    filename: Annotated[str | None, Field(description="Optional custom filename")] = None,
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    Download a file from OpenWebUI and save to workspace.
    """
    try:
        # Get authorization token from request context
        bearer_token = ctx.request_context.request.headers.get("authorization") if ctx else None
        if not bearer_token:
            return dumps({"error": {"message": "No authorization token found"}})

        logger.info(f"Downloading file {file_id} from OpenWebUI...")
        file_content = download_file_from_owui(OWUI_URL, bearer_token, file_id)

        if isinstance(file_content, dict) and "error" in file_content:
            return dumps(file_content)

        if not filename:
            metadata_url = f"{OWUI_URL}/api/v1/files/{file_id}"
            response = requests.get(metadata_url, headers={'Authorization': bearer_token})
            if response.status_code == 200:
                metadata = response.json()
                filename = metadata.get('data', {}).get('meta', {}).get('name', f'file_{file_id}')

        file_data = file_content.read()
        result = save_to_workspace(file_data, filename, subdir="uploads")

        if "error" in result:
            return dumps(result)

        logger.info(f"File saved to workspace: {result['path']}")
        return dumps({
            "success": True,
            "message": f"File '{filename}' uploaded to workspace at {result['path']}",
            "path": result['path'],
            "filename": result['filename'],
            "size": result['size']
        })

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return dumps({"error": {"message": f"Upload failed: {str(e)}"}})


@mcp.tool(
    name="export_file_to_openwebui",
    title="Export workspace file to OpenWebUI",
    description="""Upload a file from the workspace to OpenWebUI for user download.

    Use this when Python code has generated a file in the workspace and the user needs to download it.

    Args:
        filename: The name of the file in the workspace (usually in outputs/ directory)
        subdir: Subdirectory to read from ("uploads" or "outputs", default: "outputs")
    """
)
async def export_file_to_openwebui(
    filename: Annotated[str, Field(description="Name of the file to export")],
    subdir: Annotated[str, Field(description="Subdirectory: 'uploads' or 'outputs'")] = "outputs",
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    Upload a workspace file to OpenWebUI for user download.
    """
    try:
        # Get authorization token
        bearer_token = ctx.request_context.request.headers.get("authorization") if ctx else None
        if not bearer_token:
            return dumps({"error": {"message": "No authorization token found"}})

        logger.info(f"Reading file {filename} from workspace...")
        result = read_from_workspace(filename, subdir=subdir)

        if "error" in result:
            return dumps(result)

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(result["content"])
            tmp_path = tmp.name

        try:
            upload_result = upload_file_to_owui(
                OWUI_URL,
                bearer_token,
                tmp_path,
                filename
            )

            if "error" in upload_result:
                return dumps(upload_result)

            logger.info(f"File exported to OpenWebUI: {upload_result['file_id']}")
            return dumps({
                "success": True,
                "message": f"File '{filename}' exported to OpenWebUI",
                "download_link": upload_result['file_path_download'],
                "file_id": upload_result['file_id']
            })
        finally:
            import os
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return dumps({"error": {"message": f"Export failed: {str(e)}"}})


@mcp.tool(
    name="list_workspace_files",
    title="List files in workspace",
    description="""List all files in the workspace directories.

    Use this to see what files are available for processing or export.

    Args:
        subdir: Optional subdirectory filter ("uploads", "outputs", or None for all)
    """
)
async def list_workspace_files_tool(
    subdir: Annotated[str | None, Field(description="Filter by subdirectory: 'uploads', 'outputs', or None for all")] = None,
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    List files in workspace.
    """
    try:
        result = list_workspace_files(subdir=subdir)
        return dumps(result)
    except Exception as e:
        return dumps({"error": {"message": f"List failed: {str(e)}"}})


@mcp.tool(
    name="delete_workspace_file",
    title="Delete file from workspace",
    description="""Delete a file from the workspace.

    Use this to clean up temporary or processed files.

    Args:
        filename: Name of the file to delete
        subdir: Subdirectory ("uploads" or "outputs", default: "uploads")
    """
)
async def delete_workspace_file_tool(
    filename: Annotated[str, Field(description="Name of the file to delete")],
    subdir: Annotated[str, Field(description="Subdirectory: 'uploads' or 'outputs'")] = "uploads",
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    Delete a file from workspace.
    """
    try:
        result = delete_workspace_file(filename, subdir=subdir)
        return dumps(result)
    except Exception as e:
        return dumps({"error": {"message": f"Delete failed: {str(e)}"}})


@mcp.tool(
    name="move_file_to_outputs",
    title="Move file to outputs directory",
    description="""Move a file from uploads to outputs directory.

    Use this to mark a file as ready for export after processing.

    Args:
        filename: Name of the file in the uploads directory
    """
)
async def move_file_to_outputs_tool(
    filename: Annotated[str, Field(description="Name of the file to move")],
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    Move a file from uploads to outputs directory.
    """
    try:
        result = move_to_outputs(filename)
        return dumps(result)
    except Exception as e:
        return dumps({"error": {"message": f"Move failed: {str(e)}"}})


if __name__ == "__main__":
    import os
    from utils.workspace import ensure_directories
    ensure_directories()
    logger.info(f"Starting MCP file handler on port {PORT}")
    logger.info(f"Workspace directory: {WORKSPACE_DIR}")
    logger.info(f"OpenWebUI URL: {OWUI_URL}")
    mcp.run(transport="streamable-http")
