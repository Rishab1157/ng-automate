OpenHands software-agent-sdk v1.53.0 (MIT) and LangGraph 1.2.14 (MIT), vendored as plain source.
To contribute upstream: fork the original repo and open a PR from the fork.

Local changes to send upstream (each is small and keeps the old behaviour by default):

1. openhands-workspace/openhands/workspace/docker/workspace.py (DockerWorkspace)
   - New field `session_api_key`: a session key for this container only. When set, it is passed to this container
     instead of the process-wide OH_SESSION_API_KEYS_0 / SESSION_API_KEY variables, and the client uses it.
     Why: with one process-wide key, code running inside one container can read the key from /proc/1/environ and
     drive any other container's agent server (verified: HTTP 200 with the stolen key).
   - New field `extra_run_args`: extra `docker run` arguments (e.g. --shm-size).

2. openhands-tools/openhands/tools/file_editor/editor.py (FileEditor.str_replace)
   - When old_str does not appear verbatim, the editor retries with old_str.strip() but wrote new_str unchanged.
     The match starts after the file's own indentation, so the indentation the model guessed was added on top of
     it: 4 spaces in the file + 8 guessed = 12, and every "fix the indentation" retry added more (verified with
     Devstral). The fallback now also drops from new_str the leading/trailing whitespace that old_str.strip()
     removed, but only the part new_str has too (common prefix/suffix), so whitespace added on purpose is kept.
     Upstream's own test (a Markdown hard line break in new_str) still passes. Test: tests/test_file_editor_fix.py.

3. openhands-tools/openhands/tools/file_editor/utils/str_replace_hints.py (new) + a small hook in editor.py
   (FileEditor.__call__, the str_replace branch)
   - A failed str_replace also says how to fix the call, worked out from the file: for an old_str that matches
     several places, a ready-to-use old_str per place that matches only there (naming the neighbouring lines added to
     make it unique, which new_str must keep); for one that is not in the file, line numbers copied from `view`, other
     indentation (with the file's exact text), only its start or end in the file, or the most similar text; plus what
     old_str/new_str must hold when one is empty, missing or both are the same (the text format of tool calls drops
     the spaces at the start and end of every value, so a change made only there is lost), or when a value holds the
     next parameter's tag because it was not closed with `</parameter>`. New lines are added with an anchor line in
     old_str rather than `insert`, whose first line would lose its indentation in that format.
     Why: with "Please ensure it is unique" or "did not appear verbatim" alone, a small model sent the same failing
     edit 12 times in a row. Only the message changes: the edit fails exactly as before.
   - Upstream test changed: tests/tools/file_editor/test_basic_operations.py::test_str_replace_with_empty_old_str
     (the message no longer lists every line). Tests: tests/test_file_editor_hints.py (in ng-automate).

The sandboxes run THIS code: NG Automate builds its agent server image from this folder with OpenHands' own recipe
(openhands-agent-server/openhands/agent_server/docker/Dockerfile, target `source`; see docker-compose.yml,
service agent-server), and docker/sandbox-test/Dockerfile builds on it. So a change here reaches the agents after
`docker compose --profile build build agent-server sandbox-test`.
