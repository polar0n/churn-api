from pydantic import BaseModel, Field, StrictInt, StrictFloat
from typing import Literal, Annotated


YesNo = Literal["Yes", "No"]
NonNegInt = Annotated[StrictInt, Field(ge=0)]
NonNegFloat = Annotated[StrictFloat, Field(ge=0)]


class ChurnData(BaseModel):
    gender: Literal["Female", "Male"]
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    payment_method: Literal["Bank transfer", "Credit card", "Electronic check"]
    paperless_billing: YesNo
    partner: YesNo
    dependents: YesNo
    internet_service: Literal["DSL", "Fiber optic"]
    online_security: YesNo
    online_backup: YesNo
    device_protection: YesNo
    tech_support: YesNo
    streaming_tv: YesNo
    streaming_movies: YesNo
    age: NonNegInt
    tenure_months: NonNegInt
    monthly_charges: NonNegFloat
    total_charges: NonNegFloat
    num_support_tickets: NonNegInt
    num_logins_last_month: NonNegInt
    feature_usage_score: NonNegFloat
    late_payments: NonNegInt


class ChurnPrediction(BaseModel):
    prediction: list[Literal[0, 1]] = Field(..., examples=[[1]])
    probability: list[NonNegFloat] = Field(..., examples=[[0.0,1.0]])


class ChurnBatchPrediction(BaseModel):
    prediction: list[Literal[0, 1]] = Field(..., examples=[[1, 1]])
    probability: list[list[NonNegFloat]] = Field(..., examples=[[[0.0,1.0],[0.0,1.0]]])
