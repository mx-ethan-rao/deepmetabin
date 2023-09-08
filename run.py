import dotenv
import hydra
from omegaconf import DictConfig
import os.path as osp

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)
    if not osp.isabs(config['datamodule']['zarr_dataset_path']):
        config['datamodule']['zarr_dataset_path'] = osp.join(__file__.rsplit('/', 1)[0], config['datamodule']['zarr_dataset_path'])
    config['model']['zarr_dataset_path'] = config['datamodule']['zarr_dataset_path']
    if not osp.isabs(config['datamodule']['output']):
        config['datamodule']['output'] = osp.join(__file__.rsplit('/', 1)[0], config['datamodule']['output'])
    if not osp.isabs(config['model']['contignames_path']):
        config['model']['contignames_path'] = osp.join(__file__.rsplit('/', 1)[0], config['model']['contignames_path'])
    config['datamodule']['must_link_path'] = osp.join(config['datamodule']['output'], 'must_link.csv')
    config['model']['latent_save_path'] = osp.join(config['datamodule']['output'], 'latents')
    config['model']['log_path'] = config['datamodule']['output']
    config['trainer']['check_val_every_n_epoch'] = config['trainer']['max_epochs']

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
