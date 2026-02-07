"""Bash command safety judge.

Uses the LLM to evaluate whether a bash command is potentially dangerous
before execution.  If flagged, the tool returns an error instead of running.
"""

import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from axono import config

# ---------------------------------------------------------------------------
# Safety judge prompt
# ---------------------------------------------------------------------------

SAFETY_JUDGE_PROMPT = """\
You are a security analyst evaluating shell commands for safety.

A command is DANGEROUS if it could:
- Delete, overwrite, or corrupt files (rm, dd, mkfs, truncate, shred, > redirect overwriting important files)
- Modify system configuration (chmod 777, chown, editing /etc/ files)
- Install or remove system packages destructively (apt remove, pip uninstall)
- Kill or stop running processes or services (kill, killall, pkill, systemctl stop)
- Transmit sensitive data over the network (posting credentials, exfiltrating private keys)
- Execute downloaded or untrusted code (curl|bash, wget|sh, eval, exec of remote content)
- Format disks or modify partitions (fdisk, mkfs, parted)
- Shutdown or reboot the system (shutdown, reboot, init 0)
- Modify or destroy version-control history (git push --force, git reset --hard)

A command is SAFE if it only:
- Reads data (ls, cat, head, tail, find, grep, ps, df, du, top, wc, file, stat)
- Creates files/directories in user space without overwriting (mkdir, touch new files)
- Runs common development tools (git status, git log, git diff, python --version, npm list, pip list)
- Displays help or version info (--help, --version, man)
- Performs read-only network requests (curl GET, ping, dig, nslookup, wget to stdout)

Respond ONLY with a JSON object — no markdown fences, no commentary:
{"dangerous": true/false, "reason": "brief explanation"}
"""

_judge_llm = None


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = init_chat_model(
            model=config.get_model_name("instruction"),
            model_provider=config.LLM_MODEL_PROVIDER,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )
    return _judge_llm


def _parse_agent_data(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return json.dumps(content)
    return str(content)


async def judge_command(command: str) -> dict:
    """Ask the LLM whether *command* is dangerous.

    Returns ``{"dangerous": bool, "reason": str}``.
    """
    llm = _get_judge_llm()
    messages = [
        SystemMessage(content=SAFETY_JUDGE_PROMPT),
        HumanMessage(content=f"Evaluate this command:\n{command}"),
    ]
    response = await llm.ainvoke(messages)
    raw = _parse_agent_data(response.content).strip()

    # Strip markdown fences if present
    json_text = raw
    if json_text.startswith("```"):
        json_text = json_text.split("\n", 1)[1] if "\n" in json_text else json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    json_text = json_text.strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # If we can't parse the verdict, let the command through
        return {"dangerous": False, "reason": "Safety judge response unparseable"}
