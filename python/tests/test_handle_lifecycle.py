"""Tests for native handle ownership and cleanup semantics."""

from __future__ import annotations

from metahotspot._handle import OwnedHandle


def test_close_is_idempotent():
    handle = object()
    destroyed = []

    owner = OwnedHandle("dll", handle, destroyed.append)
    owner.close()
    owner.close()

    assert destroyed == [handle]


def test_owned_handle_context_manager_releases_once():
    handle = object()
    destroyed = []

    with OwnedHandle("dll", handle, destroyed.append) as owner:
        assert owner._handle is handle

    assert destroyed == [handle]
