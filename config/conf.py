import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @property
    def show_config(self) -> None:
        class_dict: dict = self.__dict__
        for attribute, value in class_dict.items():
            logging.info(f"{attribute:>20} : {value}")

    @property
    def name_str(self) -> str:
        sorted = "sort" if self.sort_ev else "unsort"
        return f"{self.normalizer}_{self.eigenvec_func}_{sorted}"
