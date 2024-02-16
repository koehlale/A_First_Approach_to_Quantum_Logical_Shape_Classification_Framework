import logging
from dataclasses import dataclass


@dataclass
class Config:
    feature_generator: str
    normalizer: str
    eigenvec_func: str
    sort_ev: bool
    number_of_bins: int
    ratio_of_train_data: int
    data_output_folder: str


def name_str(cfg: Config) -> str:
    sorted = "sort" if cfg.sort_ev else "unsort"
    return f"{cfg.normalizer}_{cfg.eigenvec_func}_{sorted}"


def show_config(confg: Config) -> None:
    class_dict: dict = confg.__dict__["_content"]
    for attribute, value in class_dict.items():
        logging.info(f"{attribute:>20} : {value}")
    return
