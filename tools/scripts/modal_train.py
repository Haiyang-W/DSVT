# # Training DSVT on NuScenes v1.0-mini with Modal

# ## Setup

from pathlib import Path
import itertools
import modal

# ### Repository Pointer
# This demo relies on (slightly) modified version of DSVT.
dsvt_repository = "https://github.com/ben-modal/DSVT-modal.git"
branch_name = "modal-train"
# After that change merged into the main repo, this can point to the regular DSVT.
# dsvt_repository = https://github.com/beijbom/DSVT.git
# branch_name = "master"
# Get the short name of the repo
dsvt = dsvt_repository.split("/")[-1].split(".")[0]

# ### NuScenes Dataset Setup
# To automatically download and setup the v1.0-mini subset of NuScenes you need to make an account here:
#       https://www.nuscenes.org/nuscenes#download
# Then upload your username and password as a [Modal Secret](https://modal.com/secrets/):
# 1) Click the Custom secret button,
# 2) name the secret `nuscenes`,
# 3) then enter two key/value pairs: {NUSCENES_USERNAME: your_email, NUSCENES_PASSWORD: your_password}.
#
# Then, the following line will import these values as environment variables:
nuscenes_secret = modal.Secret.from_name("nuscenes")

# We'll download the data into a Modal.Volume with this name:
vol_name = "example-nuscenes"
# This is the relative location within the container where this Volume will be mounted:
vol_mnt = Path("/data")
vol_data_subdir = "nuscenes"  # data subdir within the volume
CONFIG_CACHE_SUBDIR = "config-backups"  # you can save whatever you want in there


# Initialize the the Volume object (creating one if you haven't already):
nuscenes_volume = modal.Volume.from_name(vol_name, create_if_missing=True)

# ### Define the image
# Here we build all the necessary libraries to run DSVT and OpenPCDet.
# It's somewhat nuanced, with specialized versions of numpy etc., so beware of fiddling.
nuscenes_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu20.04", add_python="3.9")
    .env(
        {  # Some environment variable needed to compile the libs in the image
            "DEBIAN_FRONTEND": "noninteractive",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6",
            "CXX": "g++",
            "CC": "gcc",
        }
    )
    .apt_install(["git", "python3-opencv", "build-essential", "ninja-build", "clang"])
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --index-strategy unsafe-best-match "
        "'numpy==1.23.5' 'scikit-image<=0.21.0' "
        # Doesn't have to be this version of Torch, but need to commit to a version and stay consistent.
        "'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' 'torchaudio==2.0.2+cu118' "
        "--index-url https://download.pytorch.org/whl/cu118 "
        "--extra-index-url https://pypi.org/simple"
    )
    .run_commands(
        "uv pip install --system --no-build-isolation spconv-cu118 torch-scatter "
        "-f https://data.pyg.org/whl/torch-2.0.1+cu118.html"
    )
    .run_commands(
        "uv pip install --system tensorrt onnx pyyaml 'nuscenes-devkit==1.0.5'"
    )
    # Here we clone some repos and build them. NOTE: you could instead import your a local
    # version of the repo using Modal's `add_local_dir`:
    # https://modal.com/docs/guide/images#add-local-files-with-add_local_dir-and-add_local_file
    #
    .run_commands(f"git clone -b {branch_name} --single-branch {dsvt_repository}")
    .run_commands(f"uv pip install --system --no-build-isolation -e {dsvt}")
    .run_commands("uv pip install --system 'mmcv>=1.4.0,<2.0.0'")
    # NOTE: it might be possible to use DSVT's internally copied subset of pcdet,
    # which would save time as this takes a while:
    .run_commands("git clone https://github.com/open-mmlab/OpenPCDet.git")
    .run_commands("uv pip install --system --no-build-isolation -e OpenPCDet")
    .run_commands("uv pip install --system 'av2==0.2.0' 'kornia<0.7'")
    .run_commands(
        f"mkdir -p /OpenPCDet/data && ln -s {(vol_mnt / vol_data_subdir).as_posix()} /OpenPCDet/data/nuscenes"
    )
    .entrypoint([])
    # Add local version of training dir (assumes nothing in here needs to be rebuilt).
    # This goes last. `remote_path` must be absolute path.
    .add_local_dir("~/DSVT-modal/tools", remote_path=f"/{dsvt}/tools")
)

# Initialize the app
app = modal.App(
    "train-nuscenes",
    image=nuscenes_image,
    volumes={vol_mnt: nuscenes_volume},
)


# ## NuScenes Automated Downloading + Preprocessing
# The function `download_nuscenes` automatically downloads and preprocesses the NuScenes dataset.
# Tested with the v1.0-mini partition only -- the massive v1.0-train dataset may need special
# handling (for example, distributing across several Modal Volumes).
# NOTE: we cap max_containers with 1 here, but you could distribute the download
# and preprocessing over subsets of v1.0-train and v1.0-test.
# NOTE: it's unclear whether a GPU is necessary for preprocessing, but `mmcv` throws a lot
# of warnings if one is not available.
@app.function(
    image=nuscenes_image,
    secrets=[nuscenes_secret],
    volumes={vol_mnt: nuscenes_volume},
    timeout=60 * 60,  # preprocessing v1.0-mini from scratch takes 30-60min
    gpu="A10G",
    max_containers=1,
)
def download_nuscenes(
    volume_subdir: str,
    region: str = "us",  # or "asia"
    dataset_version: str = "v1.0-mini",  # only tested with v1.0-mini
):
    """
    Inputs:
        - volume_subdir: subdir within the volume where all this data should be downloaded/extracted
        - region: 'us' or 'asia'
        - dataset_version: string identifying which subset of the NuScenes dataset to download, see
            https://www.nuscenes.org/nuscenes#download

    Automated download code inspired by:
    https://github.com/li-xl/nuscenes-download/blob/master/download_nuscenes.py
    """
    import json
    import os
    import subprocess
    import sys
    import tarfile

    import requests
    from tqdm import tqdm

    download_dir = vol_mnt / volume_subdir
    tgz_file = download_dir / f"{dataset_version}.tgz"
    extract_dir = download_dir / dataset_version
    info_prefix = "nuscenes"

    # (0) Download .tgz from AWS:
    if not tgz_file.is_file():
        download_dir.mkdir(parents=True, exist_ok=True)
        # (1a) Get login from Modal Secret
        if not os.getenv("NUSCENES_USERNAME") or not os.getenv("NUSCENES_PASSWORD"):
            print(
                "Error: set NUSCENES_USERNAME and NUSCENES_PASSWORD in your env",
                file=sys.stderr,
            )
            sys.exit(1)

        # (1b) Log in via Cognito to get a bearer token
        print(f"Setting up Nuscenes mini dataset in `{vol_name}/{volume_subdir}`.")
        resp = requests.post(
            "https://cognito-idp.us-east-1.amazonaws.com/",
            headers={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
            },
            data=json.dumps(
                {
                    "AuthFlow": "USER_PASSWORD_AUTH",
                    "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
                    "AuthParameters": {
                        "USERNAME": os.getenv("NUSCENES_USERNAME"),
                        "PASSWORD": os.getenv("NUSCENES_PASSWORD"),
                    },
                }
            ),
        )
        resp.raise_for_status()
        token = resp.json()["AuthenticationResult"]["IdToken"]
        print("\tLogged in successfully")

        # (1c) Fetch the mini archive URL
        api = (
            f"https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1"
            f"/archives/v1.0/{tgz_file.name}?region={region}&project=nuScenes"
        )
        resp = requests.get(api, headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        download_url = resp.json()["url"]
        print(f"\tGot download URL for {tgz_file.name}")

        # (1d) Download into download_dir
        os.makedirs(download_dir, exist_ok=True)

        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with (
                open(tgz_file, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"\tDownloaded to {tgz_file}")
    else:
        print(f"\t.tgz archive found at: {tgz_file}")

    # (2) Extract the archive in-place
    n_files = len(list(extract_dir.glob("*")))
    if n_files < 3:
        print(f"\tExtracting to {extract_dir}")
        with tarfile.open(tgz_file, "r:gz") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting", unit="file"):
                tar.extract(member, path=extract_dir)
        print("\tExtraction complete")
    else:
        print(
            f"\t{n_files} found at {extract_dir}, assuming archive already extracted...!"
        )

    # (3) Generate metadata with nuScenes devkit
    pickles = [
        extract_dir / f"{info_prefix}_infos_10sweeps_train.pkl",
        extract_dir / f"{info_prefix}_dbinfos_10sweeps_withvelo.pkl",
        extract_dir / f"{info_prefix}_infos_10sweeps_val.pkl",
    ]
    if all([x.is_file() for x in pickles]):
        print("\tpickle files found!")
    else:
        print("\tGenerating devkit metadata pickles...")
        os.chdir("/OpenPCDet")
        cmd = [
            sys.executable,
            "-m",
            "pcdet.datasets.nuscenes.nuscenes_dataset",
            "--func",
            "create_nuscenes_infos",
            "--cfg_file",
            "tools/cfgs/dataset_configs/nuscenes_dataset.yaml",
            "--version",
            f"{dataset_version}",
            "--with_cam",
        ]
        subprocess.run(cmd, check=True)
    nuscenes_volume.commit()
    print(f"\tNuScenes {dataset_version} is ready in: {extract_dir}")


# ## Trainer application
# This class encapsulates the call to `train.py`. It could easily be collapsed into a simpler function,
# but we set it up as a class to demonstrate the `with_options` constructor decorator (see function below),
# which allows us to dynamically configure GPU type and count.
#
# The `train` method is the important one: this constructs and executes a call to train.py.
# The `default_nuscenes_config_setup` shows how to programmatically modify the model and data
# config so as to point at the data volume and each other.
@app.cls(
    image=nuscenes_image,
    volumes={vol_mnt: nuscenes_volume},
    timeout=24 * 60 * 60,
    cloud="aws",
)
class DSVTTrainer:
    @modal.enter()
    def dev_count(self):
        """Runs once at container startup."""
        import torch

        self.n_gpus = torch.cuda.device_count()

    def default_nuscenes_config_setup(
        self,
        tag: str = "",
        data_ver: str = "v1.0-mini",
        model_name: str = "dsvt_plain_1f_onestage_nusences",
        data_name: str = "nuscenes_dataset",
        config_save_dir="saved-configs",
        optimization_config: dict = {},
    ):
        """
        This function loads template model/data configs and modifies them to point at the data
        we created in the Modal Volume. This is meant as a placeholder depending on how
        you want to create and save configs.

        Inputs:
            tag: string added to modified configs
            data_ver: string specifying the NuScenes dataset subset version
            model_name: name of a YAML file in f"{dsvt}/tools/cfgs/dsvt_models"
            data_name: name of a YAML file in f"{dsvt}/tools/cfgs/dataset_configs"
            config_save_dir: place to save the custom configs
        """
        # Data catalogs
        from yaml import safe_dump, safe_load

        # Refresh volume:
        nuscenes_volume.reload()

        # Directories
        tools = Path(f"/{dsvt}") / "tools"
        model_configs = tools / "cfgs/dsvt_models"
        data_configs = tools / "cfgs/dataset_configs"

        # Template config defaults:
        template_model_path = model_configs / f"{model_name}.yaml"
        template_data_path = data_configs / f"{data_name}.yaml"

        # Configs we'll create & use
        savedir = vol_mnt / config_save_dir
        savedir.mkdir(exist_ok=True, parents=True)
        output_data_path = savedir / f"{tag}-data-config.yaml"
        output_model_path = savedir / f"{tag}-model-config.yaml"

        # Data catalogs
        data_dir = vol_mnt / vol_data_subdir / data_ver
        train_data = data_dir / "nuscenes_infos_10sweeps_train.pkl"
        val_data = data_dir / "nuscenes_infos_10sweeps_val.pkl"
        velo_data = data_dir / "nuscenes_dbinfos_10sweeps_withvelo.pkl"
        if not (train_data.is_file() and val_data.is_file() and velo_data.is_file()):
            raise ValueError(f"data was not found at: {data_dir}")
        ########################################################
        # (1) Edit the data config

        with open(template_data_path, "r") as f:
            data_config = safe_load(f)

        data_config["DATA_PATH"] = (vol_mnt / vol_data_subdir).as_posix()
        data_config["VERSION"] = data_ver
        data_config["INFO_PATH"] = {
            "train": [train_data.as_posix()],
            "test": [val_data.as_posix()],
        }

        data_config["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"][0]["DB_INFO_PATH"] = [
            velo_data.as_posix()
        ]

        with open(output_data_path, "w") as f:
            safe_dump(data_config, f)

        ########################################################
        # (2) Edit model config

        # (2.a) point to the new data config
        with open(template_model_path, "r") as f:
            model_config = safe_load(f)
        model_config["DATA_CONFIG"]["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"][0][
            "DB_INFO_PATH"
        ] = [velo_data.as_posix()]
        # Point model config to our new data config
        model_config["DATA_CONFIG"]["_BASE_CONFIG_"] = output_data_path.as_posix()

        # (2.b) update the optimization config
        for key, param in optimization_config.items():
            model_config["OPTIMIZATION"][key] = param
        with open(output_model_path, "w") as f:
            safe_dump(model_config, f)

        print(
            f"Using modified version of configs:"
            f"\n\tmodel: {template_model_path}"
            f"\n\tdata: {template_data_path}"
            f"\nSee modified versions at: {savedir}"
        )
        return output_model_path.as_posix()

    @modal.method()
    def train(
        self,
        tag_and_config: dict,
        data_ver: str = "v1.0-mini",
        cli_flags: dict = {},
        base_model_config: str = None,
        base_data_config: str = None,
    ):
        """
        Method that creates a command for executing train.py.
        Inputs:
            params: dict of flags passed on to train.py
        """
        import os

        # Separate inputs
        tag, opt_config = tag_and_config

        # Create the config for this experiment
        # NOTE: if you wanted to user different models or data, can pass them here
        model_config_path = self.default_nuscenes_config_setup(
            tag=tag,
            optimization_config=opt_config,
            data_ver=data_ver,
            config_save_dir=CONFIG_CACHE_SUBDIR,
        )

        # Prepare commandline arguments
        flags = " ".join([f"--{arg}={val}" for arg, val in cli_flags.items()])
        cmd = (
            f"torchrun "
            "--standalone "
            "--nnodes=1 "
            "--rdzv-backend=c10d "
            "--rdzv-endpoint=localhost:0 "
            f"--nproc_per_node={self.n_gpus} "  # one process per GPU
            f"/{dsvt}/tools/train.py --launcher pytorch "
            f"--cfg_file {model_config_path} " + flags
        )
        if tag:
            print(f"Running exp {tag} with command:\n\t{cmd}")

        # Execute
        os.chdir(f"/{dsvt}/tools")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in range(self.n_gpus)]
        )

        os.system(cmd)


def setup_data(data_ver):
    # Check if the necessary pickle files exist:
    pickles = [
        "nuscenes_infos_10sweeps_train.pkl",
        "nuscenes_infos_10sweeps_val.pkl",
        "nuscenes_dbinfos_10sweeps_withvelo.pkl",
    ]

    run_downloader = False
    try:
        paths = [
            Path(x.path).name
            for x in nuscenes_volume.listdir(f"{vol_data_subdir}/{data_ver}")
            if x.path.endswith("pkl")
        ]

        for p in pickles:
            if p not in paths:
                run_downloader = True
    except Exception:
        # If the dir does not exist in the volume, it will error out to here
        run_downloader = True

    if run_downloader:
        download_nuscenes.remote(
            volume_subdir=vol_data_subdir, region="us", dataset_version="v1.0-mini"
        )
    else:
        print("Dataset pickles found, skipping download etc.")


@app.local_entrypoint()
def main(gpu: str = "A100", n_gpus: int = 2, data_ver: str = "v1.0-mini"):
    # (0) Check for data before firing up the downloader/preprocessor container
    setup_data(data_ver=data_ver)

    # (1) Determine experiments.
    # At current this `default_nuscenes_config_setup` method sets up the BASE
    # config (data and model) so that they point to each other.
    # For the parameter sweep we will (in trainer.train):
    # 1. also dynamically edit the OPTIMIZATION field
    # 2. save with a custom filename
    # Here we specify all the training parameters we want to try

    param_lists = {
        "BATCH_SIZE_PER_GPU": [1, 2],
        "LR": [0.0001, 0.005],
        "LR_WARMUP": [True, False],
    }

    # Keys in a fixed order so the zipping is stable
    keys = list(param_lists)

    # Cartesian grid over all value lists
    params_to_sweep = [
        dict(zip(keys, combination))
        for combination in itertools.product(*(param_lists[k] for k in keys))
    ]

    # Tag each experiment EXP0, EXP1, …
    map_inputs = [(f"EXP{i}", p) for i, p in enumerate(params_to_sweep)]

    print("Experiments:", params_to_sweep)

    # Start the app and fire up that many containers

    # (1) Create the trainer app.
    trainer = DSVTTrainer.with_options(gpu=f"{gpu}:{n_gpus}")()

    # This is a constant input to the trainer.train method:
    train_kwargs = {
        "cli_flags": {"epochs": 1},  # these are forwarded to train.py
        "data_ver": data_ver,  # other kwargs for trainer.train
    }

    # (3.) Call train.py
    # for result in trainer.train.map(map_inputs, kwargs=train_kwargs):
    #     print(f"Got result from training session: {result}")
    trainer.train.spawn_map(map_inputs, kwargs=train_kwargs)

    # (4.) Use results to e.g. spawn a new experiment, etc...
