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
   - The sandbox runs the agent server's own copy of this file: the fix reaches the agents only when the sandbox
     image carries it (see docker/sandbox-test/Dockerfile).
