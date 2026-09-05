"""Argument wiring. These catch the class of bug where a flag silently does nothing."""

import pytest

from handrobot.cli import COMMANDS, build_parser


def test_every_subcommand_has_a_handler():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "no subparsers were registered"
    assert set(actions[0].choices) == set(COMMANDS)


@pytest.mark.parametrize(
    "argv",
    [
        ["info"],
        ["warmup"],
        ["warmup", "--force"],
        ["handcheck"],
        ["teleop"],
        ["collect-scripted", "--episodes", "3"],
        ["train", "--data", "data/x"],
        ["eval"],
        ["diagnose", "--checkpoint", "c.pt"],
        ["demo"],
        ["film", "--data", "data/x"],
        ["replay", "--data", "data/x"],
        ["dataset", "--data", "data/x"],
    ],
)
def test_commands_parse(argv):
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_train_flags_reach_the_config():
    args = build_parser().parse_args(
        ["train", "--data", "d", "--steps", "7", "--batch-size", "3",
         "--chunk-size", "5", "--lr", "0.5", "--no-augment"]
    )
    assert (args.steps, args.batch_size, args.chunk_size, args.lr) == (7, 3, 5, 0.5)
    assert args.no_augment


def test_diagnose_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["diagnose"])


def test_eval_defaults_to_the_scripted_expert():
    assert build_parser().parse_args(["eval"]).checkpoint is None


def test_teleop_recording_can_be_disabled():
    assert build_parser().parse_args(["teleop", "--no-record"]).no_record


def test_warmup_builds_nothing_when_everything_is_already_there(capsys, monkeypatch):
    """The step CI runs before the suite. It has to be a no-op on a warm tree,
    or every run pays for assets it already has."""
    from handrobot.cli import command_warmup

    import handrobot.retarget.reach as reach

    def refuse(*args, **kwargs):
        raise AssertionError("warmup rebuilt an asset that was already present")

    monkeypatch.setattr(reach.ReachTable, "cached", refuse)
    command_warmup(build_parser().parse_args(["warmup"]))
    out = capsys.readouterr().out
    assert "already there" in out and "ready" in out
