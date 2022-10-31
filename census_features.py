from pydantic import BaseModel, Field


class CensusFeatures(BaseModel):
    """Container for data passed to `/salary_class_inference` POST."""
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native_country")

    class Config:
        allow_population_by_field_name = True
        # Declare example as at
        # https://fastapi.tiangolo.com/tutorial/schema-extra-example/
        schema_extra = {
            "example": {
                "age": 40,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 338409,
                "education": "Masters",
                "education_num": 14,
                "marital_status": "Never-married",
                "occupation": "Exec-managerial",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Female",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }
