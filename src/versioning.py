import subprocess
import yaml
from pathlib import Path
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command with error handling.

    Args:
        cmd: Command and arguments as a list.
        description: Human-readable description of the command.

    Returns:
        True if command succeeded, False otherwise.
    """
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"  Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  Failed: {description}")
        logger.error(f"  Error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"  Command not found: {cmd[0]}")
        return False


def update_data_version(config_path: str = "configs/config.yaml") -> None:
    """Create a new DVC-tracked data version.

    Stages the processed data, commits to Git with a tag, and pushes to DVC remote.

    Args:
        config_path: Path to the configuration file.
    """
    path = Path(config_path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    current_version = config["experiment"]["data_version"]
    version_number = int(current_version.lstrip("v"))
    new_version = f"v{version_number + 1}"

    logger.info(f"Updating data version: {current_version} -> {new_version}")

    if not run_command(["dvc", "add", "processed_data"], "DVC add processed_data"):
        return

    if not run_command(["git", "add", "."], "Git add all"):
        return

    commit_msg = f"Update data version: {current_version} -> {new_version}"
    if not run_command(["git", "commit", "-m", commit_msg], "Git commit"):
        return

    tag = f"data-{new_version}"
    if not run_command(["git", "tag", "-a", tag, "-m", f"Data version {new_version}"], "Git tag"):
        return

    if not run_command(["git", "push", "origin", "main"], "Git push"):
        logger.warning("Git push failed - you may need to push manually")

    if not run_command(["dvc", "push"], "DVC push"):
        logger.warning("DVC push failed - you may need to push manually")

    with open(path, "w") as f:
        config["experiment"]["data_version"] = new_version
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Data version updated to {new_version}")


if __name__ == "__main__":
    update_data_version()
