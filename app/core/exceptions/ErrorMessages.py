class ErrorMessages:
    """Messages shown to API callers. Never put secrets or server paths in them."""

    # Auth
    INVALID_TOKEN = "Invalid or expired token"
    INVALID_TOKEN_PAYLOAD = "Token is missing a valid user_id or org_id"
    MISSING_PERMISSION = "Missing required permission"
    CROSS_ORG_FORBIDDEN = "Not allowed to act for another organization"

    # Input
    INVALID_OBJECT_ID = "{field} is not a valid id"

    # QXcel modules
    MODULE_NOT_FOUND = "QXcel module '{code}' not found"

    # Projects
    PROJECT_NOT_FOUND = "Project not found"
    PROJECT_TOO_LARGE = "Project is larger than {limit_mb} MB"

    # Archives
    INVALID_ZIP = "Not a valid zip file"
    EMPTY_ZIP = "The zip contains no files"
    TOO_MANY_FILES = "Too many files in the zip (limit {limit})"
    UNSAFE_ZIP_PATH = "Unsafe path in zip: {name}"
    ZIP_LINK_NOT_ALLOWED = "Links are not allowed in the zip: {name}"
    ZIP_TOO_LARGE_UNZIPPED = "Unzipped size is too large"

    # Git
    GIT_CONNECTION_NOT_FOUND = "Git connection not found or not enabled for NG Automate in this organization"
    GIT_ONLY_HTTPS = "Only https:// repository URLs are supported"
    GIT_CREDENTIALS_IN_URL = "Credentials belong in the git connection's token, not in the URL"
    GIT_INVALID_BRANCH = "Invalid branch name: {branch}"
    GIT_TIMEOUT = "Git took longer than {seconds} seconds"
    GIT_AUTH_FAILED = "Authentication failed: the token on this git connection is invalid or expired"
    GIT_BRANCH_NOT_FOUND = "Branch '{branch}' was not found in the repository"
    GIT_REPO_NOT_FOUND = "Repository not found, or the token has no access to it"
    GIT_HOST_UNREACHABLE = "Could not reach the Git server"
    GIT_FAILED = "Git failed: {reason}"

    # Model connections
    MODEL_CONNECTION_NOT_FOUND = "Model connection not found or not enabled for NG Automate in this organization"

    # Runs
    RUN_NOT_FOUND = "Run not found"
    RUN_NOT_ACTIVE = "The run is {status}: it no longer takes commands"
    RUN_STOPPED_BY_USER = "Stopped by the user"
    RUN_PAUSED_TOO_LONG = "Stopped: the run was paused for longer than {minutes} minutes"
    COMMAND_TEXT_REQUIRED = "A message needs text"
    COMMAND_TEXT_TOO_LONG = "A message can be at most {limit} characters"
    PROFILE_NOT_FOUND = "This project has no profile yet: start an analysis run first"

    # Test data
    TEST_DATA_NOT_FOUND = "Test data not found for this project"
    TEST_DATA_NOT_READY = "The test data is not ready (status: {status}): wait until it has been read"
    TEST_DATA_RETRY_NOT_ALLOWED = "Only test data that failed to be read can be read again"
    TEST_DATA_READ_FAILED = "The test data could not be turned into test cases: {reason}"
    TEST_DATA_REQUIRED = "A generate run needs test_data_id: upload test data for the project first"
    NO_RUNNABLE_TEST_CASES = "None of the test cases can be generated: every one has a step without a locator"

    # Sandbox
    SANDBOX_FAILED = "The sandbox could not be started: {reason}"

    # Live view
    LIVE_VIEW_NO_SANDBOX = "There is nothing to watch: the run has no test sandbox running right now"
    LIVE_VIEW_NOT_FOUND = "Live view not found or expired: ask for a new one"

    # Analyzer
    ANALYSIS_FAILED = "The analyzer could not produce a valid profile: {reason}"

    # Runner and healer
    HEAL_FAILED = "The healer stopped: {reason}"
    TEST_RUN_FAILED = "The tests could not be run: {reason}"
    GENERATION_FAILED = "The test generator stopped: {reason}"

    # Generic
    INTERNAL_ERROR = "An unexpected error occurred"
