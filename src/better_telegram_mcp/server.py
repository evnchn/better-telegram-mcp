from __future__ import annotations

import asyncio
import functools
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .backends.base import TelegramBackend
from .config import Settings
from .security import build_external_tool_result
from .tools.chats import ChatOptions, handle_chats
from .tools.config_tool import handle_config
from .tools.contacts import ContactsOptions, handle_contacts
from .tools.help_tool import handle_help
from .tools.media import MediaOptions, handle_media
from .tools.messages import MessagesArgs, handle_messages
from .utils.formatting import err, ok

# Silence httpx INFO-level request logs so the bot token in the request URL
# (https://api.telegram.org/bot<TOKEN>/...) cannot leak into stderr.
logging.getLogger("httpx").setLevel(logging.WARNING)

_backend: TelegramBackend | None = None
# Number of live lifespans holding ``_backend``. Every MCP transport session
# runs its own ``_lifespan``, but they all share one on-disk Telethon session
# file, which takes a single writer. Build one client for the first transport
# and hand it to the rest; disconnect when the last one leaves.
_backend_refs: int = 0
_backend_lock: asyncio.Lock | None = None
_backend_lock_loop: asyncio.AbstractEventLoop | None = None
_settings: Settings | None = None
_pending_auth: bool = False
_unconfigured: bool = False
_runtime_config: dict[str, int] = {
    "message_limit": 20,
    "timeout": 30,
}

# Track whether we're in multi-user HTTP mode
_multi_user_mode: bool = False
_get_current_backend = None


def _acquire_lock() -> asyncio.Lock:
    """Serialize backend connect/disconnect within the running loop.

    Rebound when the loop changes: a lock is bound to the loop that first
    awaits it, and the process runs a fresh ``asyncio.run`` per transport
    mode (and per test). A dead loop can hold no waiters, so rebinding is
    safe.
    """
    global _backend_lock, _backend_lock_loop
    loop = asyncio.get_running_loop()
    if _backend_lock is None or _backend_lock_loop is not loop:
        _backend_lock = asyncio.Lock()
        _backend_lock_loop = loop
    return _backend_lock


def get_backend() -> TelegramBackend:
    """Get the active backend.

    In multi-user HTTP mode: returns the per-user backend from ContextVar.
    In stdio/single-user mode: returns the global _backend.
    """
    if _multi_user_mode:
        global _get_current_backend
        if _get_current_backend is None:
            # ⚡ Bolt: Lazy-load get_current_backend once to avoid the overhead of
            # evaluating the import statement on every single tool invocation.
            from .transports.http import get_current_backend

            _get_current_backend = get_current_backend

        backend = _get_current_backend()
        if backend is not None:
            return backend

    if _backend is None:
        msg = "Backend not initialized. Server lifespan not started."
        raise RuntimeError(msg)
    return _backend


def get_settings() -> Settings:
    if _settings is None:
        msg = "Settings not initialized. Server lifespan not started."
        raise RuntimeError(msg)
    return _settings


def _not_ready_response() -> dict[str, Any]:
    if _unconfigured:
        return ok(
            {
                "error": "Not configured",
                "setup": {
                    "relay": "Use config(action='setup_start') to configure via browser",
                    "bot_mode": {
                        "env_var": "TELEGRAM_BOT_TOKEN",
                        "how": "Get token from @BotFather on Telegram",
                    },
                    "user_mode": {
                        "env_vars": ["TELEGRAM_PHONE"],
                        "how": "Set your phone number (API credentials have built-in defaults)",
                    },
                },
            }
        )
    return err(
        "Telegram session not authenticated. "
        "Use config(action='setup_reset') then config(action='setup_start') to reconfigure, "
        "or set TELEGRAM_PHONE in your MCP server env config."
    )


def _ensure_settings() -> Settings:
    """Initialize settings and resolve credential state."""
    settings = Settings()
    if not settings.is_configured:
        from .credential_state import resolve_credential_state

        state = resolve_credential_state()
        # If config was loaded from file, re-create Settings to pick up env vars
        if state.value == "configured":
            settings = Settings()
    return settings


def _create_backend_instance(settings: Settings) -> TelegramBackend:
    """Create a backend instance based on settings mode."""
    if settings.mode == "bot":
        from .backends.bot_backend import BotBackend

        assert settings.bot_token is not None
        return BotBackend(settings.bot_token)
    else:
        from .backends.user_backend import UserBackend

        assert settings.api_id is not None
        assert settings.api_hash is not None
        return UserBackend(settings)


def _register_hot_reload() -> None:
    """Register hot-reload callback for relay credentials."""
    from .credential_state import set_on_configured

    async def _reinit_backend_from_relay() -> None:
        global _backend, _settings, _pending_auth, _unconfigured
        _settings = _ensure_settings()
        if not _settings.is_configured:
            return
        logger.info("Hot-reloading backend from relay config ({})", _settings.mode)
        backend = _create_backend_instance(_settings)
        await backend.connect()
        _backend = backend
        _unconfigured = False
        _pending_auth = False
        logger.info("Backend hot-reloaded successfully ({})", _settings.mode)

    set_on_configured(_reinit_backend_from_relay)


async def _release_backend(owned: TelegramBackend | None) -> None:
    """Drop this lifespan's hold; the last one out disconnects.

    The bookkeeping runs before any ``await`` so a transport cancelled during
    teardown — the normal path when an HTTP client hangs up — cannot strand its
    reference and pin the backend open forever.
    """
    global _backend, _backend_refs, _unconfigured
    _backend_refs = max(0, _backend_refs - 1)
    if _backend_refs:
        return
    _unconfigured = False
    # Close both the instance this lifespan connected and whatever is published
    # now: a hot-reload may have replaced the global while we held it, and at
    # zero refs nothing needs either.
    stale: list[TelegramBackend] = []
    for backend in (owned, _backend):
        if backend is not None and all(backend is not s for s in stale):
            stale.append(backend)
    _backend = None
    async with _acquire_lock():
        for backend in stale:
            await backend.disconnect()
    if stale:
        logger.info("Disconnected from Telegram")


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[None]:
    global _backend, _backend_refs, _settings, _pending_auth, _unconfigured
    _settings = _ensure_settings()

    if not _settings.is_configured:
        if _multi_user_mode:
            # Multi-user HTTP mode: per-user backends injected via ContextVar.
            # No global backend needed — skip unconfigured state.
            logger.info("Multi-user HTTP mode: per-user backends via bearer auth.")
            try:
                yield
            finally:
                pass
            return

        _unconfigured = True
        logger.warning(
            "No Telegram credentials configured. "
            "help and config tools available; other tools will show setup instructions. "
            "Set TELEGRAM_BOT_TOKEN or TELEGRAM_API_ID + TELEGRAM_API_HASH to enable all tools."
        )

        _register_hot_reload()

        async with _acquire_lock():
            _backend_refs += 1
        try:
            yield
        finally:
            # A hot-reload may have built a backend while we were unconfigured;
            # the last transport out disconnects it.
            await _release_backend(None)
        return

    logger.info("Mode: {}", _settings.mode)
    async with _acquire_lock():
        if _backend_refs and _backend is not None:
            owned = _backend
            logger.info("Reusing Telegram backend ({} transports)", _backend_refs + 1)
        else:
            owned = _create_backend_instance(_settings)
            # Publish only once connected, so a failed connect leaves nothing
            # behind for the next transport to reuse.
            await owned.connect()
            _backend = owned
            logger.info("Connected to Telegram ({})", _settings.mode)
        _backend_refs += 1

    try:
        if _settings.mode == "user" and not await owned.is_authorized():
            _pending_auth = True
            logger.warning(
                "Session not authorized. "
                "Remove credential env vars and restart to trigger relay setup "
                "(relay handles OTP/2FA via bidirectional messaging)."
            )
        yield
    finally:
        await _release_backend(owned)


mcp = FastMCP(
    "better-telegram-mcp",
    lifespan=_lifespan,
)

# FastMCP (the SDK class) has no ``version=`` kwarg, so it defaults the
# advertised serverInfo.version to the mcp SDK's own version. Pin it to this
# package's version so initialize responses report the right value.
from better_telegram_mcp import __version__  # noqa: E402

mcp._mcp_server.version = __version__

# Register the standard `config__open_relay` MCP tool so the LLM can re-trigger
# the relay form via tool call when credentials are missing or expired. In
# stdio mode (PUBLIC_URL unset), the tool returns ``stdio_unsupported`` so
# the caller surfaces a clear "switch to HTTP mode" message. In HTTP mode,
# the tool opens ``<PUBLIC_URL>/authorize`` which renders the multi-step
# Telegram credential form (phone -> OTP -> 2FA password) already wired into
# mcp-core's HTTP OAuth AS via `custom_credential_form_html` in `run_http`.
from mcp_core.relay.tool_helpers import register_open_relay_tool  # noqa: E402

_PUBLIC_URL = os.environ.get("PUBLIC_URL")
register_open_relay_tool(mcp, "better-telegram-mcp", _PUBLIC_URL)


def _wrap_tool(tool_name: str):
    """Decorate an ``@mcp.tool`` so external Telegram content is XPIA-marked.

    The wrapped tool returns a plain ``dict``; this turns it into a
    ``CallToolResult`` so both response channels are defended: the text block
    gains ``<untrusted_{tool}_content>`` boundary tags and ``structuredContent``
    carries the envelope markers (a client reading structured output never sees
    the text block). FastMCP still derives the tool's ``outputSchema`` from the
    wrapped function's ``-> dict`` annotation, which ``functools.wraps`` keeps
    reachable via ``__wrapped__``.

    Applied only to tools that surface arbitrary users' authored content
    (message, chat, contact, media); config/help return the server's own state
    and stay plain dicts.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            result = await func(*args, **kwargs)
            return build_external_tool_result(tool_name, result)

        return wrapper

    return decorator


# --- Tools ---


@mcp.tool(
    annotations=ToolAnnotations(
        title="Telegram Messages",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@_wrap_tool("message")
async def message(
    action: str,
    chat_id: str | int | None = None,
    text: str | None = None,
    message_id: int | None = None,
    reply_to: int | None = None,
    parse_mode: str | None = None,
    from_chat: str | int | None = None,
    to_chat: str | int | None = None,
    emoji: str | None = None,
    query: str | None = None,
    limit: int = 20,
    offset_id: int | None = None,
) -> dict[str, Any]:
    """Send, edit, delete, forward, pin, react, search, and get message history.

    Actions (chat_id: "@username" | int):
    - send (chat_id, text -> reply_to, parse_mode)
    - edit (chat_id, message_id, text -> parse_mode)
    - delete (chat_id, message_id)
    - forward (from_chat, to_chat, message_id)
    - pin (chat_id, message_id)
    - react (chat_id, message_id, emoji)
    - search (query -> chat_id, limit=20)
    - history (chat_id -> limit=20, offset_id)
    """
    if _unconfigured or _pending_auth:
        return _not_ready_response()

    args = MessagesArgs(
        action=action,
        chat_id=chat_id,
        text=text,
        message_id=message_id,
        reply_to=reply_to,
        parse_mode=parse_mode,
        from_chat=from_chat,
        to_chat=to_chat,
        emoji=emoji,
        query=query,
        limit=limit,
        offset_id=offset_id,
    )
    return await handle_messages(get_backend(), args)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Telegram Chats",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@_wrap_tool("chat")
async def chat(
    action: str,
    chat_id: str | int | None = None,
    title: str | None = None,
    description: str | None = None,
    is_channel: bool = False,
    link_or_hash: str | None = None,
    user_id: int | None = None,
    demote: bool = False,
    limit: int = 50,
    topic_action: str | None = None,
    topic_id: int | None = None,
    topic_name: str | None = None,
) -> dict[str, Any]:
    """List, create, join, leave, manage members, settings, and topics.

    Actions:
    - list (-> limit=50)
    - info (chat_id)
    - create (title -> is_channel)
    - join (link_or_hash)
    - leave (chat_id)
    - members (chat_id -> limit=50)
    - admin (chat_id, user_id -> demote)
    - settings (chat_id, title|description)
    - topics (chat_id, topic_action -> topic_id, topic_name)
    """
    if _unconfigured or _pending_auth:
        return _not_ready_response()

    opts = ChatOptions(
        chat_id=chat_id,
        title=title,
        description=description,
        is_channel=is_channel,
        link_or_hash=link_or_hash,
        user_id=user_id,
        demote=demote,
        limit=limit,
        topic_action=topic_action,
        topic_id=topic_id,
        topic_name=topic_name,
    )
    return await handle_chats(get_backend(), action, opts)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Telegram Media",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@_wrap_tool("media")
async def media(
    action: str,
    chat_id: str | int | None = None,
    file_path_or_url: str | None = None,
    message_id: int | None = None,
    caption: str | None = None,
    output_dir: str | None = None,
    file_id: str | None = None,
) -> dict[str, Any]:
    """Send photos, files, voice, video, and download media from messages.

    Actions (file_path_or_url: local path or URL):
    - send_photo (chat_id, file_path_or_url -> caption)
    - send_file (chat_id, file_path_or_url -> caption)
    - send_voice (chat_id, file_path_or_url -> caption)
    - send_video (chat_id, file_path_or_url -> caption)
    - download (chat_id, message_id -> output_dir; bot mode also requires file_id)
    """
    if _unconfigured or _pending_auth:
        return _not_ready_response()

    opts = MediaOptions(
        chat_id=chat_id,
        file_path_or_url=file_path_or_url,
        message_id=message_id,
        caption=caption,
        output_dir=output_dir,
        file_id=file_id,
    )
    return await handle_media(get_backend(), action, opts)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Telegram Contacts",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
@_wrap_tool("contact")
async def contact(
    action: str,
    query: str | None = None,
    phone: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    user_id: int | None = None,
    unblock: bool = False,
) -> dict[str, Any]:
    """Manage contacts: list, search, add, and block/unblock users (user mode only).

    Actions:
    - list: Show all contacts
    - search (query): Find contacts by name
    - add (phone, first_name -> last_name)
    - block (user_id -> unblock=true)
    """
    if _unconfigured or _pending_auth:
        return _not_ready_response()

    opts = ContactsOptions(
        query=query,
        phone=phone,
        first_name=first_name,
        last_name=last_name,
        user_id=user_id,
        unblock=unblock,
    )
    return await handle_contacts(get_backend(), action, options=opts)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Telegram Config",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def config(
    action: str,
    message_limit: int | None = None,
    timeout: int | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """Server configuration and runtime settings.

    Actions (required params):
    - status: Show connection state, mode, and current config
    - set (message_limit|timeout): Update runtime limits
    - cache_clear: Clear internal caches
    - setup_status: Show credential state and relay URL
    - setup_start (-> key='force'): Start relay setup via browser
    - setup_reset: Clear saved credentials
    - setup_complete: Re-resolve credentials after relay config
    """
    # Setup actions work regardless of configured state
    match action:
        case "setup_status":
            from .credential_state import get_setup_url, get_state

            state = get_state()
            return ok(
                {
                    "state": state.value,
                    "setup_url": get_setup_url(),
                    "configured": not _unconfigured,
                    "pending_auth": _pending_auth,
                    "env_keys": [
                        k
                        for k in {"TELEGRAM_BOT_TOKEN", "TELEGRAM_PHONE"}
                        if os.environ.get(k)
                    ],
                }
            )

        case "setup_start":
            from .credential_state import CredentialState, get_state

            if get_state() == CredentialState.CONFIGURED and not (
                key and key.lower() == "force"
            ):
                return ok(
                    {
                        "status": "already_configured",
                        "message": "Already configured. Use key='force' to reconfigure.",
                    }
                )
            # Per spec 2026-05-01-stdio-pure-http-multiuser.md: stdio mode does
            # not spawn an in-process credential form. Browser-based setup is
            # the responsibility of HTTP mode; this branch tells the user how
            # to switch.
            return ok(
                {
                    "status": "stdio_unsupported",
                    "message": (
                        "Browser-based setup is HTTP-mode only. "
                        "For stdio mode, set TELEGRAM_BOT_TOKEN in your "
                        "plugin/server config (get from @BotFather). "
                        "For user-mode auth (phone+OTP), switch to HTTP mode "
                        "(see https://mcp.n24q02m.com/servers/better-telegram-mcp/setup/)."
                    ),
                }
            )

        case "setup_reset":
            from .credential_state import reset_state

            reset_state()
            return ok(
                {
                    "status": "ok",
                    "message": "Credentials cleared. Use setup_start to reconfigure.",
                }
            )

        case "setup_complete":
            from .credential_state import (
                CredentialState,
                get_state,
                resolve_credential_state,
            )

            resolve_credential_state()
            state = get_state()
            return ok(
                {
                    "status": "ok",
                    "state": state.value,
                    "message": "Credential state refreshed.",
                }
            )

    if _unconfigured:
        if action == "status":
            return ok(
                {
                    "mode": None,
                    "connected": False,
                    "authorized": False,
                    "configured": False,
                    "config": _runtime_config,
                    "setup": {
                        "bot_mode": "Set TELEGRAM_BOT_TOKEN (get from @BotFather)",
                        "user_mode": (
                            "Set TELEGRAM_PHONE"
                            " (API credentials have built-in defaults)"
                        ),
                    },
                    "hint": "Use action='setup_start' to configure via browser relay.",
                }
            )
        return _not_ready_response()
    return await handle_config(
        get_backend(),
        action,
        message_limit=message_limit,
        timeout=timeout,
    )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Telegram Help",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def help(topic: str | None = None) -> str:
    """Get full documentation for any topic.

    Topics: telegram | messages | chats | media | contacts | all (default: all)
    """
    return await handle_help(topic)


# --- Resources ---
from .resources import register_resources  # noqa: E402

register_resources(mcp)


def create_http_mcp_server() -> FastMCP:
    """Create a FastMCP server instance for multi-user HTTP mode.

    Returns the existing mcp instance (tools are shared across sessions).
    Per-user backend is injected via ContextVar per request.
    """
    global _multi_user_mode
    _multi_user_mode = True
    return mcp


async def run_http(port: int = 0) -> None:
    """Run as HTTP server with local OAuth 2.1 AS via mcp-core."""
    from mcp_core.transport.local_server import run_http_server

    from .credential_form import render_telegram_credential_form
    from .credential_state import on_step_submitted, save_credentials
    from .relay_schema import RELAY_SCHEMA

    host = os.environ.get("HOST")

    await run_http_server(
        mcp,
        server_name="better-telegram-mcp",
        relay_schema=RELAY_SCHEMA,
        port=port,
        host=host,
        on_credentials_saved=save_credentials,
        on_step_submitted=on_step_submitted,
        custom_credential_form_html=render_telegram_credential_form,
    )


def main() -> None:
    import sys

    is_http = (
        "--http" in sys.argv
        or os.environ.get("MCP_TRANSPORT") == "http"
        or os.environ.get("TRANSPORT_MODE") == "http"
    )

    if not is_http:
        # Stdio mode (default): run FastMCP stdio server directly. No bridge layer.
        # Universal MCP client compatibility (Claude Code, Cursor, VS Code Copilot, etc.).
        # Credentials come from env, the encrypted single-user config, or a saved
        # Telethon session (bot token OR phone+session). resolve_credential_state
        # consults all three and hydrates os.environ from a saved config so the
        # lifespan below picks it up; only a truly unconfigured stdio server exits.
        from .credential_state import CredentialState, resolve_credential_state

        if resolve_credential_state() != CredentialState.CONFIGURED:
            msg = (
                "[better-telegram-mcp] No Telegram credentials configured for "
                "stdio mode.\n"
                "\n"
                "Run one of:\n"
                "  better-telegram-mcp login --bot-token <token>   "
                "(bot mode, get token from @BotFather)\n"
                "  better-telegram-mcp login --phone <+number>     "
                "(user mode, interactive OTP/2FA)\n"
                "\n"
                "Or set TELEGRAM_BOT_TOKEN in your plugin/server env config.\n"
                "\n"
                "Documentation: https://mcp.n24q02m.com/servers/better-telegram-mcp/setup/\n"
            )
            sys.stderr.write(msg)
            sys.exit(1)
        mcp.run(transport="stdio")
        return

    # HTTP mode (opt-in via --http or MCP_TRANSPORT=http or TRANSPORT_MODE=http):
    # dispatch through transports/http.py so the multi-user OAuth 2.1 branch,
    # the refuse-guard for broken single-user-on-public deploys, and the
    # single-user fallback all live in one place.
    from .config import Settings
    from .transports.http import start_http

    start_http(Settings())
