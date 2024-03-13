import logging

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="settings.env")

    feature_generator: str = Field(alias="FEATURE_GENERATOR")
    normalizer: str = Field(alias="NORMALIZER")
    eigenvec_func: str = Field(alias="EIGENVEC_FUNCTION")
    sort_ev: bool = Field(
        alias="SORTING_EIGENVALUES",
    )
    number_of_bins: int = Field(alias="NUMBER_OF_BINS", ge=1)
    ratio_of_train_data: int = Field(alias="RATIO_OF_TRAIN_DATA", ge=0, le=100)
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

    @classmethod
    def _check_str(cls, check: str, reference: tuple[str]) -> str:
        if check not in reference:
            raise ValueError(
                f" The chosen option '{check}' is not valid. Use one of {reference} instead."
            )
        return check

    @field_validator("normalizer")
    @classmethod
    def check_correct_normalizer(cls, v: str) -> str:
        possible_input = ["max", "mean"]
        return cls._check_str(check=v, reference=possible_input)

    @field_validator("eigenvec_func")
    @classmethod
    def check_correct_eigenvec_func(cls, v: str) -> str:
        possible_input = ["eigh", "eig"]
        return cls._check_str(check=v, reference=possible_input)

    @field_validator("feature_generator")
    @classmethod
    def check_correct_feature_generator(cls, v: str) -> str:
        possible_input = ["D1"]
        return cls._check_str(check=v, reference=possible_input)
