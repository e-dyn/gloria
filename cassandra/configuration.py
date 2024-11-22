# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:37:43 2024

@author: BeKa
"""

### --- Module Imports --- ###
# Standard Library
from pathlib import Path
import json

# Third Party
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from typing import Literal

# Inhouse Packages
from cassandra.models import MODEL_MAP


### --- Global Constants Definitions --- ###



### --- Class and Function Definitions --- ###


class DataConfig(BaseModel):
    data_source: str
    sampling_period: str
    timestamp_name: str
    

class MetricConfig(BaseModel):
    metric_name: str
    model: Literal[tuple(MODEL_MAP.keys())]
    dtype_kind: Literal[tuple('biuf')]
    
    @field_validator('dtype_kind')
    @classmethod
    def validate_model_kind(
            cls,
            dtype_kind: str,
            info: FieldValidationInfo
        ) -> str:
        allowed_types = list(MODEL_MAP[info.data['model']].kind)
        if dtype_kind not in allowed_types:
            type_list = ', '.join([f"'{s}'" for s in allowed_types])
            raise TypeError(
                f"dtype_kind was set to '{dtype_kind}', but must be any of"
                f" {type_list} for selected model '{info.data['model']}'."
            )
        return dtype_kind
    
    
class CassandraConfig(BaseModel):
    n_changepoints: int = Field(ge = 0, default = 25)
    changepoint_range: float = Field(gt = 0, lt = 1, default = 0.8)
    seasonality_mode: Literal['additive', 'multiplicative'] = 'additive'
    seasonality_prior_scale: float = Field(gt = 0, default = 10.0)
    changepoint_prior_scale: float = Field(gt = 0, default = 0.05)
    interval_width: float = Field(gt = 0, lt = 1, default = 0.8)
    uncertainty_samples: int = Field(ge = 0, default = 1000)
    optimize_mode: Literal['MAP', 'MLE'] = 'MAP'
    sample: bool = True
    
    
class RunConfig(BaseModel):
    data_config: DataConfig
    metric_config: MetricConfig
    cassandra_config: CassandraConfig
    
    def to_json(self, path: Path):
        config_dict = self.dict()
        with open(path, "w") as file: 
            json.dump(config_dict, file, indent = 4)
            
    @classmethod
    def load_json(cls, path: Path):
        with open(path, 'r') as file:
            data = json.load(file)
        return cls(**data)
        

### --- Main Script --- ###
if __name__ == "__main__":
    basepath = Path(__file__).parent
    
    
    data_config = DataConfig(data_source='', sampling_period='1d', timestamp_name='ds')
    metric_config = MetricConfig(name='y', model='binomial constant n', dtype_kind='u')
    cassandra_config = CassandraConfig()
    
    run_config = RunConfig(data_config=data_config, metric_config=metric_config, cassandra_config=cassandra_config)