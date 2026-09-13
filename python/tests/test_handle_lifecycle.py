"""Tests for native handle ownership and cleanup semantics."""

from __future__ import annotations

from metahotspot._handle import OwnedHandle


class _HandleOwner(OwnedHandle):
    pass


def test_owned_handle_initializes_with_complete_handle_state():
    handle = object()
    destroyed = []

    owner = _HandleOwner("dll", handle, destroyed.append)
    owner.close()
    owner.close()

    assert destroyed == [handle]


def test_owned_handle_context_manager_releases_once():
    handle = object()
    destroyed = []

    with _HandleOwner("dll", handle, destroyed.append) as owner:
        assert owner._handle is handle

    assert destroyed == [handle]
