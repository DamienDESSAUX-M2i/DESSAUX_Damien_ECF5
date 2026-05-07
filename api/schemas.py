from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class TelcoCustomerFeatures(BaseModel):
    """Define input features for Telco churn prediction."""

    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="Whether the customer is a senior citizen (0 or 1)"
    )
    Partner: Literal["Yes", "No"] = Field(..., description="Whether the customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether the customer has dependents")

    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed")

    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has phone service"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Multiple phone lines status"
    )

    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Type of internet service"
    )

    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(...)
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(...)
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(...)
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(...)
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(...)
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(...)

    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(...)
    PaperlessBilling: Literal["Yes", "No"] = Field(...)
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(...)

    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")

    @model_validator(mode="after")
    def validate_business_rules(self) -> TelcoCustomerFeatures:
        """Validate cross-field business constraints."""
        # Phone service coherence
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("MultipleLines must be 'No phone service' when PhoneService is 'No'")

        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError(
                "MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'"
            )

        # Internet service coherence
        internet_features = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        ]

        if self.InternetService == "No":
            if any(feature != "No internet service" for feature in internet_features):
                raise ValueError(
                    "All internet-related features must be 'No internet service'"
                    "when InternetService is 'No'"
                )

        else:
            if any(feature == "No internet service" for feature in internet_features):
                raise ValueError(
                    "Internet-related features cannot be 'No internet service'"
                    "when InternetService is active"
                )

        # Charges consistency
        if self.tenure == 0 and self.TotalCharges > 0:
            raise ValueError("TotalCharges must be 0 when tenure is 0")

        if self.TotalCharges < self.MonthlyCharges:
            raise ValueError("TotalCharges must be greater than or equal to MonthlyCharges")

        return self


class PredictionRequest(BaseModel):
    """Define prediction request payload."""

    instances: list[TelcoCustomerFeatures] = Field(
        ..., max_length=100, description="Inputs features for Telco churn prediction"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "instances": [
                    {
                        "gender": "Female",
                        "SeniorCitizen": 0,
                        "Partner": "Yes",
                        "Dependents": "No",
                        "tenure": 12,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "Fiber optic",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "Yes",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "Yes",
                        "StreamingMovies": "No",
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 70.5,
                        "TotalCharges": 845.5,
                    }
                ]
            }
        }
    }


Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class PredictionResponse(BaseModel):
    """Define prediction response payload."""

    churns: list[str] = Field(
        ..., max_length=100, description="Predicted churn label (true or false)"
    )
    probabilities: list[Probability | None] = Field(
        ..., max_length=100, description="Prediction probabilities scores"
    )


class HealthResponse(BaseModel):
    """Define health check response."""

    status: Literal["healthy", "unhealthy"]
    model: str = Field(..., description="Model name")
    version: int | None = Field(..., description="Model version")
