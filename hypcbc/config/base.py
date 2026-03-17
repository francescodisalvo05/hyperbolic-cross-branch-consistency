from __future__ import annotations
from typing import Any, Dict, Type, TypeVar, Union
from typing import get_type_hints
from dataclasses import dataclass, fields
from abc import ABC
from pathlib import Path


T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig(ABC):
    """Base configuration class with validation and serialization."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import fields
        from pathlib import Path
        
        result = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if isinstance(value, BaseConfig):
                result[field_info.name] = value.to_dict()
            elif isinstance(value, Path):
                result[field_info.name] = str(value)
            else:
                result[field_info.name] = value
        return result
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create config from dictionary with type conversion."""
        type_hints = get_type_hints(cls)  # Resolves forward refs correctly
        converted_data = {}

        for field in fields(cls):
            key = field.name
            field_type = type_hints.get(key, field.type)

            if key not in data:
                continue

            value = data[key]

            # Handle Optional/Union types
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                non_none_types = [t for t in field_type.__args__ if t is not type(None)]
                field_type = non_none_types[0] if non_none_types else field_type

            # Recursively convert nested BaseConfig fields
            if isinstance(value, dict) and isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                converted_data[key] = field_type.from_dict(value)

            # Convert string to Path
            elif field_type is Path and isinstance(value, str):
                converted_data[key] = Path(value)

            else:
                converted_data[key] = value

        return cls(**converted_data)