"""Rules that every agent's task message shares."""

# Models that write their tool calls as text (LLM_NATIVE_TOOL_CALLING=false) tend to wrap parameter values in quotes,
# and the quotes then become part of the value: a glob for "**/*.java", quotes included, finds nothing.
PLAIN_TOOL_VALUES_RULE = (
    'Write tool parameter values as plain text, without surrounding quotes: a glob pattern is **/*.java, not "**/*.java".'
)
