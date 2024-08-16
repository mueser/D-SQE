from dataclasses import dataclass, field, fields
from abc import ABC

def format_value(value):
    if isinstance(value, bool):
        return str(value)
    elif isinstance(value, int):
        return f"{value:_d}"
    elif isinstance(value, float):
        return f"{value:.6g}"
    else:
        return str(value)

@dataclass
class ParamData(ABC):
    name: str = field(default=None)
    class_name: str = field(init=False)

    def __post_init__(self):
        self.class_name = self.__class__.__name__.lower()
        self.write_to_file()

    def write_to_file(self):
        with open("params.out", "a") as f:
            f.write(f"# {self.class_name}\n")
            for field in fields(self):
                if field.name not in ['name', 'class_name']:
                    value = getattr(self, field.name)
                    if value is not None:  # Only write non-None values
                        formatted_value = format_value(value)
                        padded_name = f"{field.name:<15}"  # Pad name to at least 15 characters
                        f.write(f"{padded_name}\t{formatted_value}\n")
            f.write(f"\n")

    @classmethod
    def from_dict(cls, data: dict):
        field_names = [f.name for f in fields(cls) if f.name not in ['class_name']]
        return cls(**{k: v for k, v in data.items() if k in field_names})
