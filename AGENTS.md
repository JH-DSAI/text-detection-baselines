# Instructions for agents

* Include tests where appropriate. Avoid testing the details of CLI output formatting unless that
  formatting is essential to correctness. The results table is rendered by `rich` and truncates
  headers at 80 columns, so assertions on its contents are brittle.

## Testing the CLI

Use `click.testing.CliRunner` (see the `runner` fixture in `tests/conftest.py`). It exercises the
real parameter parsing, param types, and exit codes, and a full evaluation run costs about 10ms.
Do not call a command's `.callback` directly: that bypasses everything click does, so the test
proves nothing about the command's wiring and has to hand-supply converted values and defaults.

Reach for `subprocess` only when the entry point itself or import-time state is what is under
test — module-as-script wiring, the `__main__` guard, or anything resolved when the dataset
registry is populated on import. `tests/test_cli_smoke.py` has the one example.

Two rules keep in-process runs fast, and exist because ignoring them once produced tests that
looked like they had hung:

* **Never pass `-a` / `--all` / `--all-models` in an in-process test.** It selects `smollm2`,
  which downloads model weights when constructed. Under `CliRunner` the output is captured, so a
  multi-minute download shows no progress at all. Pass explicit `--model` names instead.
* **Keep in-process datasets small.** Use a handful of rows written to `tmp_path`, or the bundled
  200-row `demo` dataset. Anything registered at runtime needs the `clean_registry` fixture, since
  the dataset registry is module-level global state that in-process tests would otherwise leak
  between them.

Tests have a 60-second timeout (`pytest-timeout`), so a genuine stall fails with a traceback
rather than requiring you to kill the terminal.
