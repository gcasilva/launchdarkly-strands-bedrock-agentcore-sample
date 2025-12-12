import os
import json
import logging
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import ldclient
from ldclient import Context
from ldclient.config import Config
from ldai.client import LDAIClient, AIConfig

from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Create a custom logger for bedrock agentcore observability
bedrock_logger = logging.getLogger('bedrock_agentcore_observability')
bedrock_logger.setLevel(logging.INFO)

# Add a handler that outputs in a format bedrock agentcore might expect
class BedrockObservabilityHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            # Format log record for bedrock agentcore observability
            log_entry = {
                "timestamp": int(record.created * 1000),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            # Output to stdout for bedrock agentcore to capture
            print(f"BEDROCK_OBSERVABILITY: {json.dumps(log_entry)}", file=sys.stdout, flush=True)
        except Exception:
            # Fallback to standard logging if custom format fails
            super().emit(record)

bedrock_handler = BedrockObservabilityHandler(sys.stdout)
bedrock_logger.addHandler(bedrock_handler)

# Test logging at module level to ensure it works in bedrock agentcore
logger.info("[DEBUG] Module loaded successfully - logging is working")
bedrock_logger.info("Bedrock observability logger initialized")

# ─────────────────────────────────────────────────────────────────────────────
# LaunchDarkly Configuration
# ─────────────────────────────────────────────────────────────────────────────
LD_SERVER_KEY = os.getenv("LD_SERVER_KEY")
LD_AI_CONFIG_ID = os.getenv("LD_AI_CONFIG_ID", "multi-agent-strands-agentcore-2")
LD_PROJECT_KEY = os.getenv("LD_PROJECT_KEY")

# Debug environment variables
logger.info(f"[DEBUG] Environment variables:")
logger.info(f"[DEBUG] LD_SERVER_KEY present: {bool(LD_SERVER_KEY)}")
logger.info(f"[DEBUG] LD_AI_CONFIG_ID: {LD_AI_CONFIG_ID}")
logger.info(f"[DEBUG] LD_PROJECT_KEY: {LD_PROJECT_KEY}")

# Initialize LaunchDarkly SDK
if LD_SERVER_KEY:
    try:
        logger.info("[DEBUG] Attempting to initialize LaunchDarkly SDK")
        ldclient.set_config(Config(LD_SERVER_KEY))
        logger.info("[DEBUG] LaunchDarkly config set")
        ld_ai_client = LDAIClient(ldclient.get())
        logger.info("[DEBUG] LDAIClient created")
        logger.info(f"[LD] LaunchDarkly initialized for project '{LD_PROJECT_KEY}'")
        logger.info(f"[LD] AI Config ID: '{LD_AI_CONFIG_ID}'")
    except Exception as ld_init_error:
        logger.error(f"[DEBUG] Error initializing LaunchDarkly: {ld_init_error}")
        ld_ai_client = None
else:
    ld_ai_client = None
    logger.warning("[LD] Warning: LD_SERVER_KEY not set – running without LaunchDarkly")

# Fallback AI config when LD is unreachable or disabled
FALLBACK_AI_CONFIG = AIConfig(enabled=False)

# ─────────────────────────────────────────────────────────────────────────────
# Bedrock AgentCore Application
# ─────────────────────────────────────────────────────────────────────────────
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    """
    AI agent entrypoint with comprehensive logging for bedrock agentcore observability.
    """
    try:
        logger.info("[DEBUG] Starting invoke function")
        logger.info(f"[AGENT] Received payload: {json.dumps(payload, indent=2)}")
        
        # Log request payload for bedrock agentcore observability
        try:
            app.logger.info(f"Request payload: {json.dumps(payload)}")
            logger.info("[DEBUG] Successfully logged to app.logger")
        except AttributeError:
            logger.info("[DEBUG] app.logger not available")
        except Exception as e:
            logger.error(f"[DEBUG] Error logging to app.logger: {e}")
        
        print(f"REQUEST_PAYLOAD: {json.dumps(payload)}", file=sys.stdout, flush=True)
        bedrock_logger.info(f"Request payload: {json.dumps(payload)}")
        logger.info("[DEBUG] Logged request payload")
        
        user_message = payload.get("prompt", "Hello! How can I help you today?")
        logger.info(f"[AGENT] User message: {user_message}")
        
        # ─────────────────────────────────────────────────────────────────────────
        # Retrieve AI Config from LaunchDarkly
        # ─────────────────────────────────────────────────────────────────────────
        system_prompt = None
        model_name = None
        config = None
        tracker = None

        if ld_ai_client:
            logger.info(f"[LD] Retrieving AI Config '{LD_AI_CONFIG_ID}'")
            
            # Build LaunchDarkly context
            context_key = payload.get("user_id", "anonymous-user")
            logger.info(f"[LD] Building context with key: {context_key}")
            
            from ldclient import Context
            context = Context.builder(context_key).build()
            logger.info(f"[LD] Context built successfully")

            # Custom variables to interpolate in AI Config messages (optional)
            variables = {"user_message": user_message}
            logger.info(f"[LD] Using variables: {json.dumps(variables)}")

            try:
                config, tracker = ld_ai_client.config(
                    LD_AI_CONFIG_ID,
                    context,
                    FALLBACK_AI_CONFIG,
                    variables,
                )
                
                logger.info(f"[LD] AI Config retrieved successfully")
                logger.info(f"[LD] Config enabled: {config.enabled}")
                
                if config.enabled:
                    # Extract model name from config
                    if config.model and hasattr(config.model, 'name'):
                        model_name = config.model.name
                        logger.info(f"[LD] Model from config: {model_name}")
                    
                    # Extract system prompt from messages
                    if hasattr(config, 'messages') and config.messages:
                        logger.info(f"[LD] Found {len(config.messages)} messages in config")
                        for msg in config.messages:
                            if msg.role == "system":
                                system_prompt = msg.content
                                logger.info(f"[LD] Found system message: {system_prompt[:100]}...")
                                break
                else:
                    logger.warning(f"[LD] AI Config '{LD_AI_CONFIG_ID}' is disabled")
                
            except Exception as e:
                logger.error(f"[LD] Error retrieving AI Config: {str(e)}")
                config = FALLBACK_AI_CONFIG
        else:
            logger.info("[LD] No LaunchDarkly client available, using defaults")
        
        # ─────────────────────────────────────────────────────────────────────────
        # Initialize Agent with LaunchDarkly configuration
        # ─────────────────────────────────────────────────────────────────────────
        agent_kwargs = {}
        
        if system_prompt:
            agent_kwargs["system_prompt"] = system_prompt
            logger.info(f"[AGENT] Using system prompt from LD config")
        
        if model_name:
            agent_kwargs["model"] = model_name
            logger.info(f"[AGENT] Using model from LD config: {model_name}")
        
        logger.info(f"[AGENT] Initializing agent with kwargs: {json.dumps({k: v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v for k, v in agent_kwargs.items()})}")
        logger.info("[DEBUG] About to initialize agent")
        print("DEBUG: About to initialize agent", flush=True)
        
        agent = Agent(**agent_kwargs)
        logger.info("[DEBUG] Agent initialized successfully")
        print("DEBUG: Agent initialized successfully", flush=True)
        
        logger.info("[DEBUG] About to call agent")
        print("DEBUG: About to call agent", flush=True)
        
        result = agent(user_message)
        logger.info("[DEBUG] Agent call completed")
        print("DEBUG: Agent call completed", flush=True)
        
        # Track success metrics back to LaunchDarkly
        if tracker and config and config.enabled:
            logger.info("[LD] Tracking success metrics")
            tracker.track_success()
        
        # Create response payload
        response_payload = {"result": result.message}
        
        # Multiple logging approaches to ensure bedrock agentcore captures the response
        logger.info(f"[AGENT] Response payload: {json.dumps(response_payload)}")
        
        try:
            app.logger.info(f"Response payload: {json.dumps(response_payload)}")
            logger.info("[DEBUG] Successfully logged response to app.logger")
        except AttributeError:
            logger.info("[DEBUG] app.logger not available for response")
        except Exception as e:
            logger.error(f"[DEBUG] Error logging response to app.logger: {e}")
        
        print(f"RESPONSE_PAYLOAD: {json.dumps(response_payload)}", file=sys.stdout, flush=True)
        print(f"RESPONSE_PAYLOAD: {json.dumps(response_payload)}", file=sys.stderr, flush=True)
        
        bedrock_logger.info(f"Response payload: {json.dumps(response_payload)}")
        
        # Structured log for bedrock agentcore
        structured_log = {
            "event": "agent_response",
            "response_payload": response_payload,
            "timestamp": int(time.time() * 1000)
        }
        logger.info(json.dumps(structured_log))
        
        logger.info("[DEBUG] About to return response payload")
        print("DEBUG: About to return response payload", flush=True)
        
        return response_payload
        
    except Exception as e:
        logger.error(f"[DEBUG] Exception in invoke function: {str(e)}")
        import traceback
        logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
        print(f"DEBUG: Exception occurred: {str(e)}", flush=True)
        
        # Return a basic error response
        error_response = {"error": f"Error: {str(e)}"}
        logger.error(f"[DEBUG] Returning error response: {json.dumps(error_response)}")
        
        try:
            app.logger.error(f"Error response payload: {json.dumps(error_response)}")
        except:
            pass
            
        return error_response


if __name__ == "__main__":
    logger.info("[DEBUG] Starting application")
    app.run()
