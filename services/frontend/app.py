"""Product Intelligence System - Chatbot Assistant Frontend."""

import asyncio
from pathlib import Path

import streamlit as st

from config import get_settings
from components.chat import render_chat_interface

settings = get_settings()

# Page configuration
st.set_page_config(
    page_title="Product Intelligence Assistant",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="auto",
)


def load_custom_css():
    """Load custom CSS for chat interface."""
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }

    /* Header styling */
    .chat-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }

    .chat-header h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .chat-header p {
        color: #666;
        font-size: 1rem;
    }

    /* Product card in chat */
    .product-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #e9ecef;
    }

    .product-card:hover {
        border-color: #007bff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .product-title {
        font-weight: 600;
        color: #333;
        margin-bottom: 4px;
    }

    .product-price {
        color: #28a745;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .product-rating {
        color: #ffc107;
    }

    .product-brand {
        color: #6c757d;
        font-size: 0.85rem;
    }

    /* Suggestions styling */
    .suggestions-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }

    .suggestion-chip {
        background: #e7f3ff;
        color: #0066cc;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        cursor: pointer;
        border: 1px solid #cce5ff;
    }

    .suggestion-chip:hover {
        background: #cce5ff;
    }

    /* Intent badge */
    .intent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 8px;
    }

    .intent-search { background: #d4edda; color: #155724; }
    .intent-compare { background: #cce5ff; color: #004085; }
    .intent-recommend { background: #fff3cd; color: #856404; }
    .intent-greeting { background: #e2e3e5; color: #383d41; }
    .intent-help { background: #d1ecf1; color: #0c5460; }

    /* Message metadata */
    .message-meta {
        font-size: 0.75rem;
        color: #999;
        margin-top: 4px;
    }

    /* Stacked button fix */
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application - Chat-only interface."""
    load_custom_css()

    # Header
    st.markdown("""
    <div class="chat-header">
        <h1>🛒 Product Intelligence Assistant</h1>
        <p>Ask me anything about products - search, compare, get recommendations, and more!</p>
    </div>
    """, unsafe_allow_html=True)

    # Render chat interface
    render_chat_interface()

    # Footer with capabilities hint
    with st.expander("💡 What can I help you with?", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🔍 Product Discovery**
            - "Find wireless headphones under $100"
            - "Show me best-rated laptops"

            **📊 Comparison**
            - "Compare Sony vs Bose headphones"
            - "Which is better for gaming?"
            """)
        with col2:
            st.markdown("""
            **💡 Recommendations**
            - "Recommend headphones for travel"
            - "Best budget earbuds"

            **💬 General Questions**
            - "What can you do?"
            - "Help me find a product"
            """)


if __name__ == "__main__":
    main()
