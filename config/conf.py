import logging
from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="settings.env", extra="allow")

    feature_generator: str = Field(
        alias="FEATURE_GENERATOR",
    )
    normalizer: str = Field(alias="NORMALIZER")
    eigenvec_func: str = Field(alias="EIGENVEC_FUNCTION")
    sort_ev: bool = Field(alias="SORTING_EIGENVALUES")
    number_of_bins: int = Field(alias="NUMBER_OF_BINS")
    ratio_of_train_data: int = Field(alias="RATIO_OF_TRAIN_DATA")
    data_output_folder: str = Field(alias="DATA_OUTPUT_FOLDER")
