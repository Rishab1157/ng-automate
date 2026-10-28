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

4. openhands-tools/openhands/tools/file_editor/editor.py (FileEditor)
   - `create` makes the missing folders of the new file's path (after the same path checks as before). Why: the
     write goes through a temp file in the file's folder, so in a new folder every create failed with
     "[Errno 2] No such file or directory: '.../.Name.feature.xxxx.tmp'" and gpt-4o sent the same create 6 times.
   - Clearer errors (what happened, why, what to try): a path that does not exist (names the nearest folder that
     exists), create on an existing file or folder, a folder where a file is expected, a bad view_range or
     insert_line (says how many lines the file has), and a failed write (names the file, not the temp file).
     Only the messages change. Upstream tests changed: test_error_handling.py, test_basic_operations.py.

5. openhands-tools/openhands/tools/terminal (constants.py, terminal/terminal_session.py, impl.py)
   - Exit codes 127 (command not found) and 126 (cannot run) say what they mean and what to check.
   - Timeout messages say what to do. A pooled terminal (tmux, what the sandboxes use) stops a command that is
     still running before it runs the next one there (TerminalExecutor._prepare_pooled_session), so upstream's
     advice to wait with '' or send keys only failed with "No previous running command ..." there; a pooled
     session now says to run the command again with a larger `timeout`. Only the messages change.
     Upstream test helper changed: tests/tools/terminal/conftest.py::get_no_change_timeout_suffix.

6. openhands-sdk/openhands/sdk/agent/repeated_failures.py (new) + a hook in agent.py (_ActionBatch.emit, which
   both the sync and the async loop use for every executed call, parallel ones included)
   - Per conversation, the failed results in a row of each call (tool name + its arguments as sorted JSON, long
     strings as their sha1) are counted; a success of that call starts again. From the 3rd failure on, the error
     the model reads ends with a note that repeating the exact call will not help. Nothing is skipped, blocked or
     stopped, and the StuckDetector is unchanged (for an ObservationEvent the note is in extended_content, so the
     observation it compares is the same). Tests: tests/sdk/agent/test_repeated_failures.py.

7. openhands-tools/openhands/tools (grep/impl.py, grep/definition.py, glob/impl.py, glob/definition.py, utils/__init__.py)
   - grep `include` matched only the bare file name with fnmatch, so "**/*.java", "src/test/java/**/*.java" and the
     tool's own example form "*.{ts,tsx}" always gave "No files found" (65 of 92 grep calls in real runs; the agents
     then guessed imports). include now follows ripgrep's -g rules for every backend: without "/" it matches the file
     name in any folder (as before), with "/" the path relative to the searched folder ("**" = any folders), and
     {a,b} is expanded (_expand_braces, max 1000 patterns). ripgrep itself no longer gets -g: it read globs with "/"
     relative to the server's own folder, so they found nothing there either.
   - The system grep (what the agent image has: no ripgrep) searched the regex as a basic regex, so "public|artifactId"
     was taken literally. It now runs `grep -P` (Perl-compatible, like Python's re). When grep or ripgrep cannot take
     the regex (exit code 2: look-arounds in ripgrep, Python-only escapes such as \N{...}), the Python search answers
     instead of a false "No files found"; so does a regex that spans lines. The pattern goes after -e (a pattern such
     as "-x" was read as an option), and the Python search now matches ^/$ per line and skips binary files, like grep.
     On Windows, Git's grep (an MSYS program) expanded wildcards in the regex and lost its backslashes: MSYS=noglob.
   - A relative `path` started at the server's own folder ("not a valid directory"); it now starts at the working
     directory (grep and glob). Absolute paths work as before.
   - Actionable messages when nothing matched: include names a folder (use path), include is a list ("*.java,*.xml":
     use braces, with the exact include to send), include with "/" (what it is relative to); a file passed as path;
     glob pattern naming a folder (use "folder/**/*"). glob's Python fallback expands braces ("**/*.{json,yaml}", the
     tool's own example, found nothing), an absolute pattern plus path now searches the pattern's folder, and a
     ripgrep error is reported instead of "No files found".
     Tests: tests/tools/grep/test_grep_include_and_path.py, tests/tools/glob/test_glob_braces_and_path.py,
     tests/tools/test_brace_expansion.py.

The sandboxes run THIS code: NG Automate builds its agent server image from this folder with OpenHands' own recipe
(openhands-agent-server/openhands/agent_server/docker/Dockerfile, target `source`; see docker-compose.yml,
service agent-server), and docker/sandbox-test/Dockerfile builds on it. So a change here reaches the agents after
`docker compose --profile build build agent-server` and THEN `docker compose --profile build build sandbox-test`:
two commands, in this order. One command for both builds them at the same time, so sandbox-test gets the old
agent-server image as its base.
