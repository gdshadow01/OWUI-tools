import subprocess
import tempfile
import os
from fastapi import FastAPI
from pydantic import BaseModel

WORKSPACE = os.environ.get("WORKSPACE_DIR", "/workspace")

app = FastAPI(
    title="MCP Python Tool",
    openapi_url="/openapi.json"
)

class PythonRequest(BaseModel):
    code: str

class BashRequest(BaseModel):
    command: str


@app.post("/exec/python")
def run_python(req: PythonRequest):
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=WORKSPACE,
        delete=False
    ) as f:
        f.write(req.code)
        filename = f.name

    result = subprocess.run(
        ["python", filename],
        capture_output=True,
        text=True,
        cwd=WORKSPACE
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr
    }


@app.post("/exec/bash")
def run_bash(req: BashRequest):
    result = subprocess.run(
        req.command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=WORKSPACE
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr
    }


@app.get("/files/list")
def list_files():
    files = []
    for root, _, filenames in os.walk(WORKSPACE):
        for name in filenames:
            files.append(os.path.relpath(os.path.join(root, name), WORKSPACE))
    return {"files": files}
