"""Enhanced chat interface component for Product Intelligence Assistant."""

import asyncio
from typing import Any

import streamlit as st

from utils.api_client import get_api_client


def _run_async(coro):
    """Run async coroutine safely in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def render_chat_interface():
    """Render the enhanced chat interface."""
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "pending_suggestion" not in st.session_state:
        st.session_state.pending_suggestion = None
    if "message_counter" not in st.session_state:
        st.session_state.message_counter = 0

    # Display chat history
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            _render_message(message, msg_idx)

    # Handle suggestion click (if any)
    if st.session_state.pending_suggestion:
        suggestion = st.session_state.pending_suggestion
        st.session_state.pending_suggestion = None
        _process_user_message(suggestion)

    # Chat input
    if prompt := st.chat_input("Ask about products..."):
        _process_user_message(prompt)


def _process_user_message(prompt: str):
    """Process a user message and get response."""
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display response
    with st.chat_message("assistant"):
        with st.spinner("Processing your request... (complex queries may take up to a few minutes)"):
            response = _run_async(_get_chat_response(prompt))

        if response.get("error"):
            st.error(f"Sorry, I encountered an error: {response['error']}")
            assistant_message = {
                "role": "assistant",
                "content": f"Error: {response['error']}",
            }
        else:
            assistant_message = _display_response(response)

        # Update session ID
        if response.get("session_id"):
            st.session_state.session_id = response["session_id"]

        # Save to history
        st.session_state.messages.append(assistant_message)

    # Force rerun to show updated chat
    st.rerun()


def _render_message(message: dict, msg_idx: int):
    """Render a message from history."""
    if message["role"] == "user":
        st.markdown(message["content"])
    else:
        # Render assistant message
        st.markdown(message.get("content", ""))

        # Render products if present
        products = message.get("products", [])
        if products:
            _render_products(products)

        # Don't render suggestions for historical messages (only for latest)
        # This avoids duplicate key issues and cluttered UI


def _display_response(response: dict) -> dict:
    """Display response and return message dict for history."""
    response_text = response.get("response", response.get("response_text", ""))
    intent = response.get("intent", "")
    products = response.get("products", [])
    suggestions = response.get("suggestions", [])
    confidence = response.get("confidence", 0)
    execution_time = response.get("execution_time_ms", 0)

    # Display main response
    st.markdown(response_text)

    # Display products
    if products:
        _render_products(products)

    # Display comparison if present
    comparison = response.get("comparison")
    if comparison and comparison.get("summary"):
        with st.expander("📊 Comparison Details", expanded=False):
            st.markdown(comparison["summary"])
            if comparison.get("winner"):
                st.success(f"**Winner:** {comparison['winner']}")
            if comparison.get("winner_reason"):
                st.info(comparison["winner_reason"])

    # Display suggestions with unique key prefix
    if suggestions:
        st.session_state.message_counter += 1
        _render_suggestions(suggestions, st.session_state.message_counter)

    # Display metadata
    if intent or execution_time:
        meta_parts = []
        if intent:
            meta_parts.append(f"Intent: {intent}")
        if execution_time:
            meta_parts.append(f"Response time: {execution_time:.0f}ms")
        st.caption(" | ".join(meta_parts))

    return {
        "role": "assistant",
        "content": response_text,
        "products": products,
        "suggestions": suggestions,
        "intent": intent,
    }


def _render_products(products: list[dict]):
    """Render product cards in chat."""
    if not products:
        return

    st.markdown("---")
    st.markdown(f"**Found {len(products)} products:**")

    # Display up to 5 products
    for i, product in enumerate(products[:5]):
        _render_product_card(product, i)


def _render_product_card(product: dict, index: int):
    """Render a single product card."""
    title = product.get("title", "Unknown Product")
    price = product.get("price")
    rating = product.get("rating") or product.get("stars")
    brand = product.get("brand")
    availability = product.get("availability", "")

    # Truncate long titles
    display_title = title[:80] + "..." if len(title) > 80 else title

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**{index + 1}. {display_title}**")
        meta_parts = []
        if brand:
            meta_parts.append(f"Brand: {brand}")
        if availability:
            meta_parts.append(availability)
        if meta_parts:
            st.caption(" | ".join(meta_parts))

    with col2:
        if price:
            st.markdown(f"**${price:.2f}**")
        if rating:
            stars = "⭐" * int(float(rating))
            st.markdown(f"{stars} ({float(rating):.1f})")


def _render_suggestions(suggestions: list[str], msg_id: int):
    """Render clickable suggestion chips."""
    if not suggestions:
        return

    st.markdown("---")
    st.markdown("**💡 You might also want to ask:**")

    # Create columns for suggestions
    cols = st.columns(min(len(suggestions), 3))
    for i, suggestion in enumerate(suggestions[:3]):
        with cols[i]:
            # Use msg_id to ensure unique keys across all messages
            if st.button(suggestion, key=f"sug_{msg_id}_{i}", use_container_width=True):
                st.session_state.pending_suggestion = suggestion
                st.rerun()


async def _get_chat_response(query: str) -> dict[str, Any]:
    """Get response from chat API using /chat/v2 endpoint."""
    import httpx

    client = get_api_client()
    try:
        return await client.chat_v2(query, st.session_state.session_id)
    except httpx.TimeoutException:
        return {
            "error": "Request timed out. Complex queries may take longer to process. Please try again or simplify your query."
        }
    except httpx.HTTPStatusError as e:
        return {"error": f"Server error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}
