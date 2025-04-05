from config import config
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id = config.repo_to_download,
    local_dir = config.path_to_save,
    local_dir_use_symlinks = False,
)
