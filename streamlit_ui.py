# --- START OF FILE streamlit_ui.py ---

# Future import MUST be the very first line of code (after comments)
from __future__ import annotations

# Apply nest_asyncio patch AFTER the future import
import nest_asyncio

nest_asyncio.apply()

# --- Now the rest of your imports ---
import asyncio
import os
import json
from typing import Any, List, Union, Literal, TypedDict

import streamlit as st
import logfire
from supabase import create_client, Client

# Import necessary components from pydantic-ai
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter,
)

# Import the Gemini-powered agent and its dependencies class
# --- Make sure 'mumbai_expert' is the correct variable name defined in pydantic_ai_expert.py ---
from pydantic_ai_expert import mumbai_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client, define functions, etc.
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    st.error(
        "Supabase URL or Key not found in environment variables. Please check your .env file."
    )
    st.stop()

supabase_client: Client = create_client(supabase_url, supabase_key)

logfire.configure(send_to_logfire="never")

SessionMessageType = Union[ModelRequest, ModelResponse]


# Corrected display_message_part function (removed duplicate blocks)
def display_message_part(part: Any):
    if not hasattr(part, "part_kind"):
        print(f"Warning: Skipping display of item without 'part_kind': {type(part)}")
        return

    part_kind = getattr(part, "part_kind", "unknown")  # Get kind safely

    if part_kind == "system-prompt":
        # Hidden by default
        pass
    elif part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)
    elif part_kind == "tool-call":
        # Display tool calls - ensure this is not commented out if you want visibility
        tool_name = getattr(part, "tool_name", "unknown_tool")
        tool_args = getattr(part, "tool_args", {})
        with st.expander(f"⚙️ Calling Tool: `{tool_name}`"):
            st.json(tool_args)
    elif part_kind == "tool-return":
        # Display tool returns - ensure this is not commented out if you want visibility
        tool_name = getattr(part, "tool_name", "unknown_tool")
        content = getattr(part, "content", "")
        with st.expander(f"Tool Result: `{tool_name}`"):
            content_str = str(content)
            max_len = 500  # Truncate long results
            display_content = content_str[:max_len] + (
                "..." if len(content_str) > max_len else ""
            )
            # Display results as code block for better formatting of potentially structured text
            st.markdown(f"```\n{display_content}\n```")
    elif part_kind == "retry-prompt":
        # Hidden by default
        pass
    else:
        # Fallback for any other unexpected part kinds
        with st.chat_message("assistant"):
            st.markdown(f"_(Unsupported part type: {part_kind})_")


async def run_agent_with_streaming(user_input: str):
    deps = PydanticAIDeps(supabase=supabase_client)
    st.session_state.messages.append(
        ModelRequest(parts=[UserPromptPart(content=user_input)])
    )
    message_placeholder = st.empty()
    full_response = ""
    try:
        # --- Use the imported agent variable name ('mumbai_expert' in this case) ---
        async with mumbai_expert.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[
                :-1
            ],  # History *before* current input
        ) as result:
            partial_text = ""
            final_response_message = None
            # Stream text response chunk by chunk
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(
                    partial_text + "▌"
                )  # Show typing indicator
            # Update placeholder with final complete text
            message_placeholder.markdown(partial_text)
            full_response = partial_text

            # Process and store the new messages (tool calls/returns, final response)
            new_messages = result.new_messages()
            # Filter out the user prompt message if the agent includes it in the output
            filtered_new_messages = [
                msg
                for msg in new_messages
                if not (
                    isinstance(msg, ModelRequest)
                    and hasattr(msg, "parts")
                    and any(
                        isinstance(p, UserPromptPart) and p.content == user_input
                        for p in msg.parts
                    )
                )
            ]
            # Add the filtered new messages to the session state history
            st.session_state.messages.extend(filtered_new_messages)

            # Double-check if the final streamed text response was added correctly
            # This handles cases where stream_text finishes but the final response object
            # might not be explicitly in new_messages.
            if not (
                st.session_state.messages
                and isinstance(st.session_state.messages[-1], ModelResponse)
                and hasattr(st.session_state.messages[-1], "parts")
                and any(
                    isinstance(p, TextPart) and p.content == full_response
                    for p in st.session_state.messages[-1].parts
                )
            ):
                if full_response:
                    print("DEBUG: Manually adding final ModelResponse to history.")
                    st.session_state.messages.append(
                        ModelResponse(parts=[TextPart(content=full_response)])
                    )
    except Exception as e:
        st.error(f"An error occurred during agent execution: {e}")
        st.exception(e)  # Display detailed exception info in the UI


async def main():
    st.set_page_config(layout="wide")
    st.title("🤖 Mumbai Agent")
    st.write(
        "Ask questions about Mumbai Construction docs or the ingested PDF content (e.g., NBC 2016)."
    )

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages: List[SessionMessageType] = []

    # Display existing messages from history
    for msg in st.session_state.messages:
        if hasattr(msg, "parts") and isinstance(msg.parts, list):
            for part in msg.parts:
                display_message_part(part)

    # Get user input from chat interface
    user_input = st.chat_input("Your question:")

    # Process user input if provided
    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        # Show spinner and run the agent's streaming response logic
        with st.spinner("Thinking..."):
            await run_agent_with_streaming(user_input)
        # Streamlit should automatically rerun after the await completes,
        # refreshing the chat display with new messages. Explicit st.rerun() is usually not needed.


if __name__ == "__main__":
    # Run the main async function using asyncio.run()
    asyncio.run(main())

# --- END OF FILE streamlit_ui.py ---
