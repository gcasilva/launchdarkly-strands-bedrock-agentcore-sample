Sample agent using LaunchDarkly AI Configs SDK + Strands SDK deployed on Amazon Bedrock Agentcore.

Pre-requisites:
Python
AWS CLi

To run:

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

Install the following required packages:

strands-agents - The Strands Agents SDK (https://strandsagents.com/latest/)
bedrock-agentcore - The Amazon Bedrock AgentCore SDK for building AI agents
bedrock-agentcore-starter-toolkit - The Amazon Bedrock AgentCore starter toolkit

pip install bedrock-agentcore strands-agents bedrock-agentcore-starter-toolkit

Verify installation of Bedrock Agentcore SDK:
agentcore --help

Configure your Bedrock Agentcore Agent:
agentcore configure -e my_agent.py -r <YOUR-AWS-REGION>

Deploy your Bedrock Agentcore Agent:
agentcore launch

Test your deployed agent:
agentcore invoke '{"prompt": "Hello"}'
