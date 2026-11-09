# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery must fence stale owners, survive lost replies, and preserve tokens."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from vllm.satellite.state import (
    RecoveryCosts,
    RequestSnapshot,
    StateConflict,
    StateJournal,
)


def snapshot():
    return RequestSnapshot("r", "model@revision", (1, 2, 3), (4, 5), 4, 20, "{}")


def prepare(journal):
    journal.register("r", "source")
    journal.append_output("r", "source", 0, 0, (4, 5))
    digest = journal.put_snapshot(snapshot())
    journal.prepare("handoff", "r", "source", 0, "target", digest)
    return digest


def test_committed_handoff_survives_lost_reply_and_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    journal = StateJournal(path)
    digest = prepare(journal)
    journal.ready("handoff", "target", digest)
    committed = journal.commit("handoff")
    journal.close()
    recovered = StateJournal(path)
    assert recovered.commit("handoff") == committed
    assert (committed.owner, committed.epoch) == ("target", 1)
    assert recovered.snapshot(digest) == snapshot()
    with pytest.raises(StateConflict):
        recovered.append_output("r", "source", 0, 2, (6,))
    recovered.append_output("r", "target", 1, 2, (6,))
    assert recovered.tokens("r") == (4, 5, 6)
    recovered.close()


def test_terminal_output_and_completion_are_committed_atomically(tmp_path):
    path = tmp_path / "journal.sqlite"
    journal = StateJournal(path)
    journal.register("r", "source")
    journal.append_output("r", "source", 0, 0, (4, 5), finished=True)
    journal.close()
    journal = StateJournal(path)
    assert journal.ownership("r").phase == "FINISHED"
    assert journal.tokens("r") == (4, 5)
    assert journal.append_output("r", "source", 0, 0, (4, 5), finished=True) == 2
    with pytest.raises(StateConflict):
        journal.append_output("r", "source", 0, 2, (6,), finished=True)
    journal.close()


def test_old_output_retry_cannot_terminate_a_longer_history(tmp_path):
    journal = StateJournal(tmp_path / "journal.sqlite")
    journal.register("r", "source")
    journal.append_output("r", "source", 0, 0, (4, 5))
    journal.append_output("r", "source", 0, 2, (6,))
    with pytest.raises(StateConflict):
        journal.append_output("r", "source", 0, 0, (4, 5), finished=True)
    assert journal.ownership("r").phase == "RUNNING"
    journal.close()


@pytest.mark.parametrize("ready", [False, True])
def test_abort_restores_source_with_new_generation(tmp_path, ready):
    journal = StateJournal(tmp_path / "state.sqlite")
    digest = prepare(journal)
    if ready:
        journal.ready("handoff", "target", digest)
    owner = journal.abort("handoff")
    assert (owner.owner, owner.epoch, owner.phase) == ("source", 1, "RUNNING")
    assert journal.abort("handoff") == owner
    with pytest.raises(StateConflict):
        journal.commit("handoff")
    with pytest.raises(StateConflict):
        journal.append_output("r", "source", 0, 2, (6,))
    journal.append_output("r", "source", 1, 2, (6,))
    assert journal.tokens("r", 1) == (5, 6)
    journal.close()


@pytest.mark.parametrize("ready", [False, True])
def test_cancellation_during_handoff_prevents_either_owner_resuming(tmp_path, ready):
    journal = StateJournal(tmp_path / "state.sqlite")
    digest = prepare(journal)
    if ready:
        journal.ready("handoff", "target", digest)
    journal.cancel("r")
    for operation in (journal.commit, journal.abort):
        with pytest.raises(StateConflict):
            operation("handoff")
    for owner, epoch in (("source", 0), ("target", 1)):
        with pytest.raises(StateConflict):
            journal.append_output("r", owner, epoch, 2, (6,))
    assert journal.ownership("r").phase == "CANCELLED"
    journal.close()


def test_output_retries_are_deduplicated_and_conflicts_rejected(tmp_path):
    journal = StateJournal(tmp_path / "state.sqlite")
    journal.register("r", "source")
    assert journal.append_output("r", "source", 0, 0, (4, 5)) == 2
    assert journal.append_output("r", "source", 0, 0, (4, 5)) == 2
    for offset, tokens in ((0, (9,)), (1, (5,)), (3, (6,))):
        with pytest.raises(StateConflict):
            journal.append_output("r", "source", 0, offset, tokens)
    assert journal.tokens("r") == (4, 5)
    journal.close()


def test_two_destinations_cannot_acquire_the_same_request(tmp_path):
    path = tmp_path / "state.sqlite"
    initial = StateJournal(path)
    initial.register("r", "source")
    initial.append_output("r", "source", 0, 0, (4, 5))
    digest = initial.put_snapshot(snapshot())
    initial.close()
    barrier = Barrier(2)

    def attempt(target):
        journal = StateJournal(path)
        barrier.wait(timeout=5)
        try:
            journal.prepare(target, "r", "source", 0, target, digest)
            return True
        except StateConflict:
            return False
        finally:
            journal.close()

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(attempt, ("a", "b")))
    assert sorted(results) == [False, True]


def test_incomplete_or_wrong_snapshot_cannot_be_committed(tmp_path):
    journal = StateJournal(tmp_path / "state.sqlite")
    journal.register("r", "source")
    journal.append_output("r", "source", 0, 0, (4, 5))
    digest = journal.put_snapshot(replace(snapshot(), output_token_ids=(4,)))
    with pytest.raises(StateConflict):
        journal.prepare("h", "r", "source", 0, "target", digest)
    assert journal.ownership("r").phase == "RUNNING"
    journal.close()


def test_storage_corruption_is_detected_before_restore(tmp_path):
    path = tmp_path / "state.sqlite"
    journal = StateJournal(path)
    digest = journal.put_snapshot(snapshot())
    with sqlite3.connect(path) as storage:
        storage.execute("UPDATE snapshots SET payload='{}' WHERE digest=?", (digest,))
    with pytest.raises(StateConflict, match="Corrupt"):
        journal.snapshot(digest)
    journal.close()


@pytest.mark.parametrize(
    ("costs", "deadline", "choice"),
    [
        (RecoveryCosts(0.1, 2, 3), 5, "keep"),
        (RecoveryCosts(None, 2, 3), 5, "transfer"),
        (RecoveryCosts(None, 10, 3), 5, "recompute"),
        (RecoveryCosts(None, 10, 3), 1, "wait"),
    ],
)
def test_recovery_policy_accounts_for_state_movement_cost(costs, deadline, choice):
    assert costs.choose(deadline) == choice
