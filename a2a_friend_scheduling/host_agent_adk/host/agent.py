import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
#DEbug: remove this import: add car_tools
from .pickleball_tools import (
    book_pickleball_court,
    list_court_availabilities,
)
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()


class HostAgent:
    """The Host agent."""

    def __init__(
        self,
    ):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        """Asynchronously initializes the components of the Host agent."""
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description}) 
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No friends found"

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: List[str],
    ):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

#DEBUG Host Agent Config
    def create_agent(self) -> Agent:
        """ Creates ADK Agent from the HostAgent's configuration.
            
        """
        return Agent(
            model="gemini-2.0-flash", #check Model!!
            name="Host_Agent",
            instruction=self.root_instruction,
            description="This Host agent orchestrates scheduling pickleball with friends.", #change discription
            tools=[
                self.send_message,
                book_pickleball_court,
                list_court_availabilities,
            ], #Remove book_pickleball_court and list_court_availabilities add Car Agent and Carfax Agent tools
        )   
#Debug: Model Prompt: 
    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
        **Role:** You are the Host Agent, an expert scheduler for pickleball games. Your primary function is to coordinate with friend agents to find a suitable time to play and then book a court.

        **Core Directives:**

        *   **Initiate Planning:** When asked to schedule a game, first determine who to invite and the desired date range from the user.
        *   **Task Delegation:** Use the `send_message` tool to ask each friend for their availability.
            *   Frame your request clearly (e.g., "Are you available for pickleball between 2024-08-01 and 2024-08-03?").
            *   Make sure you pass in the official name of the friend agent for each message request.
        *   **Analyze Responses:** Once you have availability from all friends, analyze the responses to find common timeslots.
        *   **Check Court Availability:** Before proposing times to the user, use the `list_court_availabilities` tool to ensure the court is also free at the common timeslots.
        *   **Propose and Confirm:** Present the common, court-available timeslots to the user for confirmation.
        *   **Book the Court:** After the user confirms a time, use the `book_pickleball_court` tool to make the reservation. This tool requires a `start_time` and an `end_time`.
        *   **Transparent Communication:** Relay the final booking confirmation, including the booking ID, to the user. Do not ask for permission before contacting friend agents.
        *   **Tool Reliance:** Strictly rely on available tools to address user requests. Do not generate responses based on assumptions.
        *   **Readability:** Make sure to respond in a concise and easy to read format (bullet points are good).
        *   Each available agent represents a friend. So Bob_Agent represents Bob.
        *   When asked for which friends are available, you should return the names of the available friends (aka the agents that are active).
        *   When get

        **Today's Date (YYYY-MM-DD):** {datetime.now().strftime("%Y-%m-%d")}

        <Available Agents>
        {self.agents}
        </Available Agents>
        """

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Streams the agent's response to a given query.
        """
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "The host agent is thinking...",
                }

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to a remote friend agent."""
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        # Simplified task and context ID management
        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(message_request)
        print("send_response", send_response)

        if not isinstance(
            send_response.root, SendMessageSuccessResponse
        ) or not isinstance(send_response.root.result, Task):
            print("Received a non-success or non-task response. Cannot proceed.")
            return

        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)

        resp = []
        if json_content.get("result", {}).get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    resp.extend(artifact["parts"])
        return resp

### agent tools and available agent defined
def _get_initialized_host_agent_sync():
    """Synchronously creates and initializes the HostAgent."""

    async def _async_main():
        # Hardcoded URLs for the friend agents (change to car agent and carfax agent urls)
        friend_agent_urls = [
            "http://localhost:10002",  # Karley's Agent 
            "http://localhost:10003",  # Nate's Agent
            "http://localhost:10004",  # Kaitlynn's Agent
        ]

        print("initializing host agent")
        hosting_agent_instance = await HostAgent.create(
            remote_agent_addresses=friend_agent_urls
        )
        print("HostAgent initialized")
        return hosting_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize HostAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing HostAgent within an async function in your application."
            )
        else:
            raise


root_agent = _get_initialized_host_agent_sync() 
