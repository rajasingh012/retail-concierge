"""RetailConcierge — Streamlit web UI for the AMD AI DevMaster 2026 hackathon."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailConcierge",
    page_icon="🛒",
    layout="wide",
)

# ── init the agent once (cached) ─────────────────────────────────────────────
@st.cache_resource
def get_agent():
    from infrastructure.agent_tools import build_tools
    from infrastructure.chat_clients import build_chat_client
    from infrastructure.database import ABOCatalogRepository
    from use_cases.shopping_agent import (
        CatalogEvidenceTracker,
        build_shopping_agent,
    )

    provider = os.getenv("RETAIL_PROVIDER", "deepseek")
    model = os.getenv("RETAIL_MODEL", "deepseek-v4-flash")
    db_path = Path(os.getenv("RETAIL_DB", "./retail_catalog.db"))

    repo = ABOCatalogRepository(db_path)
    client = build_chat_client(provider, model)
    tracker = CatalogEvidenceTracker()
    catalog_tools = build_tools(repo, catalog_tracker=tracker)
    agent = build_shopping_agent(client, catalog_tools, tracker=tracker, provider=provider)

    stats = repo.stats()
    return agent, repo, stats


agent, repo, stats = get_agent()

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 RetailConcierge")
    st.caption("AMD AI DevMaster 2026 — Track 2: Agentic AI")
    st.divider()
    st.metric("Catalog", f"{stats['listings']:,} products")
    st.metric("Product types", f"{stats['product_types']:,}")
    st.divider()
    model = os.getenv("RETAIL_MODEL", "deepseek-v4-flash")
    st.caption(f"Model: `{model}`")
    st.caption(f"Provider: `{os.getenv('RETAIL_PROVIDER', 'deepseek')}`")
    if st.button("🔄 New Session"):
        st.session_state.messages = []
        st.session_state.chips = []
        st.rerun()

# ── session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chips" not in st.session_state:
    st.session_state.chips = []
if "session" not in st.session_state:
    from use_cases.shopping_agent import CatalogEvidenceTracker
    st.session_state.session = agent.create_session()
    st.session_state.tracker = CatalogEvidenceTracker()

def _render_card(item: dict) -> None:
    rank = item.get("rank", "?")
    title = item.get("title_en", "Untitled")
    brand = item.get("brand_en", "")
    url = item.get("product_url", "")
    item_id = item.get("item_id", "")

    with st.container(border=True):
        col1, col2 = st.columns([1, 9])
        with col1:
            st.markdown(f"### #{rank}")
        with col2:
            if brand:
                st.markdown(f"**{brand}**")
            if url:
                st.markdown(f"[{title}]({url}) :arrow_upper_right:")
            else:
                st.markdown(f"**{title}**")
            st.caption(f"`{item_id}`")

        pros = item.get("why_it_fits", [])
        cons = item.get("trade_offs", [])
        if pros or cons:
            pcol, ccol = st.columns(2)
            with pcol:
                if pros:
                    st.caption("#### ✅ Pros")
                    for p in pros:
                        st.caption(f"• {p}")
            with ccol:
                if cons:
                    st.caption("#### ⚠️ Trade-offs")
                    for c in cons:
                        st.caption(f"• {c}")

# ── title ────────────────────────────────────────────────────────────────────
st.title("RetailConcierge")
st.caption("Ask me anything — I'll search the ABO catalog and recommend products.")

# ── refinement chips (above the input) ──────────────────────────────────────
if st.session_state.chips:
    cols = st.columns(len(st.session_state.chips))
    for i, chip in enumerate(st.session_state.chips):
        if cols[i].button(chip.label, key=f"chip_{i}"):
            st.session_state.messages.append({"role": "user", "content": chip.instruction})
            st.rerun()

# ── chat input ───────────────────────────────────────────────────────────────
prompt = st.chat_input("What are you looking for?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# ── display message history ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), dict):
            rec = msg["content"]
            st.markdown(f"**{rec.get('recommendation', 'Here are my recommendations:')}**")
            for item in rec.get("ranked", []):
                _render_card(item)
            if rec.get("assumptions"):
                with st.expander("Assumptions"):
                    for a in rec["assumptions"]:
                        st.caption(f"• {a}")
            if rec.get("refinement_chips"):
                st.caption("Refine your search:")
        elif msg["role"] == "assistant":
            st.write(msg["content"])
        elif msg["role"] == "user":
            st.write(msg["content"])

# ── handle new user input ────────────────────────────────────────────────────
if prompt:
    st.session_state.chips = []
    with st.spinner("Searching the catalog..."):
        try:
            from use_cases.shopping_agent import (
                finalized_candidates_from_response,
                structured_recommendation_from_response,
            )
            st.session_state.tracker.reset()
            response = asyncio.run(agent.run(prompt, session=st.session_state.session))
            rec = structured_recommendation_from_response(response)

            if rec is not None:
                payload = {
                    "kind": rec.kind,
                    "ranked": [item.model_dump() for item in rec.ranked],
                    "assumptions": rec.assumptions,
                    "notes": rec.notes,
                    "refinement_chips": [c.model_dump() for c in rec.refinement_chips],
                    "recommendation": rec.recommendation,
                }
                st.session_state.messages.append({"role": "assistant", "content": payload})
                st.session_state.chips = list(rec.refinement_chips)
            else:
                text = response.text or ""
                if not text.strip():
                    finalized = finalized_candidates_from_response(response)
                    if finalized:
                        text = f"Found {len(finalized)} items but could not format results."
                    else:
                        text = "I need more context to help you. What type of product are you looking for?"
                st.session_state.messages.append({"role": "assistant", "content": text})
        except Exception as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Something went wrong: {exc}"}
            )
    st.rerun()


# ── product card renderer ────────────────────────────────────────────────────

