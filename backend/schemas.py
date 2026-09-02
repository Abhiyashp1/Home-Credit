from pydantic import BaseModel,Field
class User_input(BaseModel):
    Id:int=Field(gt=0)
    Income:float=Field(ge=0)
    Credit:float=Field(ge=0)
    Annuity:float=Field(ge=0)
    Days_employed:float=Field(le=0)
    Family_members:int=Field(gt=0)
    Good_price:float=Field(gt=0)
class ShapImpact(BaseModel):
    feature:str
    impact:float
class Output(BaseModel):
    probability : float
    risk_band: str
    shap_values:list[ShapImpact]
class Application_detail(BaseModel):
    Income:float=Field(ge=0)
    Credit:float=Field(ge=0)
    Annuity:float=Field(ge=0)
    Days_employed:float=Field(le=0)
    Family_members:int=Field(gt=0)
    Good_price:float=Field(gt=0)
