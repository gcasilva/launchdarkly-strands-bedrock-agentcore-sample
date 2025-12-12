import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import ldclient
from ldclient import Context
from ldclient.config import Config
from ldai.client import LDAIClient, AIConfig

from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

# ─────────────────────────────────────────────────────────────────────────────
# LaunchDarkly Configuration
# ─────────────────────────────────────────────────────────────────────────────
LD_SERVER_KEY = os.getenv("LD_SERVER_KEY")
LD_AI_CONFIG_ID = os.getenv("LD_AI_CONFIG_ID", "multi-agent-strands")
LD_PROJECT_KEY = os.getenv("LD_PROJECT_KEY")

# Initialize LaunchDarkly SDK
if LD_SERVER_KEY:
    ldclient.set_config(Config(LD_SERVER_KEY))
    ld_ai_client = LDAIClient(ldclient.get())
    print(f"[LD] LaunchDarkly initialized for project '{LD_PROJECT_KEY}'")
else:
    ld_ai_client = None
    print("[LD] Warning: LD_SERVER_KEY not set – running without LaunchDarkly")

# Fallback AI config when LD is unreachable or disabled
FALLBACK_AI_CONFIG = AIConfig(enabled=False)

# ─────────────────────────────────────────────────────────────────────────────
# Bedrock AgentCore Application
# ─────────────────────────────────────────────────────────────────────────────
app = BedrockAgentCoreApp()


def build_context(payload: dict) -> Context:
    """
    Build a LaunchDarkly context from the incoming payload.
    You can expand this to include user attributes for targeting.
    """
    # Use a unique key from payload or fallback to anonymous
    context_key = payload.get("user_id", "anonymous-user")
    builder = Context.builder(context_key)

    # Add any additional attributes from payload for targeting
    if "email" in payload:
        builder.set("email", payload["email"])
    if "name" in payload:
        builder.set("name", payload["name"])

    return builder.build()


@app.entrypoint
def invoke(payload):
    """
    AI agent entrypoint with LaunchDarkly AI Config integration.

    The AI Config controls:
      - Whether the agent is enabled/disabled
      - System prompt / instructions
      - Model configuration (name, parameters)
    """
    user_message = payload.get("prompt", "Hello! How can I help you today?")

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieve AI Config from LaunchDarkly
    # ─────────────────────────────────────────────────────────────────────────
    system_prompt = None
    model_name = None
    config = None
    tracker = None

    if ld_ai_client:
        context = build_context(payload)

        # Custom variables to interpolate in AI Config messages (optional)
        variables = {
            "user_message": user_message,
        }

        config, tracker = ld_ai_client.config(
            LD_AI_CONFIG_ID,
            context,
            FALLBACK_AI_CONFIG,
            variables,
        )

        if not config.enabled:
            # AI Config is disabled – you may choose to block or use defaults
            print(f"[LD] AI Config '{LD_AI_CONFIG_ID}' is disabled for this context")
        else:
            # Extract system prompt from the first message (if provided)
            if config.messages:
                # Look for a system message
                for msg in config.messages:
                    if msg.role == "system":
                        system_prompt = msg.content
                        break
                # If no explicit system message, use the first message content
                if system_prompt is None and config.messages:
                    system_prompt = config.messages[0].content

            # Extract model name from config
            if config.model and config.model.name:
                model_name = config.model.name

            print(f"[LD] Using AI Config '{LD_AI_CONFIG_ID}' | model={model_name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Initialize Agent with LaunchDarkly-provided configuration
    # ─────────────────────────────────────────────────────────────────────────
    agent_kwargs = {}

    if system_prompt:
        agent_kwargs["system_prompt"] = system_prompt

    if model_name:
        # strands Agent accepts a model parameter for Bedrock model ID
        agent_kwargs["model"] = model_name

    agent = Agent(**agent_kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Execute Agent and track metrics
    # ─────────────────────────────────────────────────────────────────────────
    try:
        result = agent(user_message)

        # Track success metrics back to LaunchDarkly
        if tracker and config and config.enabled:
            tracker.track_success()

        return {"result": result.message}

    except Exception as e:
        # Track error metrics
        if tracker and config and config.enabled:
            tracker.track_error()
        raise


if __name__ == "__main__":
    app.run()
