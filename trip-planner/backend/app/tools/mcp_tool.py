from langchain.tools import BaseTool
import subprocess
import json

class MCPTool(BaseTool):
    name: str
    description: str
    server_command: list[str]
    env: dict[str, str]

    def call(self, tool_name: str, arguments: dict) -> str:
        payload = {
            "action": "call_tool",
            "tool_name": tool_name,
            "arguments": arguments
        }

        proc = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )

        stdout, _ = proc.communicate(json.dumps(payload))
        return stdout

    def list_tools(self) -> list[dict]:
        payload = {
            "action": "list_tools"
        }

        proc = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )

        stdout, stderr = proc.communicate(json.dumps(payload))

        if stderr:
            raise RuntimeError(stderr)

        return json.loads(stdout).get("tools", [])