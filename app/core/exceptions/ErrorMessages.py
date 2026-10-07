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
    PROFILE_NOT_FOUND = "This project has no profile yet: start an analysis run first"

    # Sandbox
    SANDBOX_FAILED = "The sandbox could not be started: {reason}"

    # Analyzer
    ANALYSIS_FAILED = "The analyzer could not produce a valid profile: {reason}"

    # Generic
    INTERNAL_ERROR = "An unexpected error occurred"
