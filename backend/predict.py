from fastapi import APIRouter
from model_store import get_models
from schemas import User_input,Output,ShapImpact
from inference import get_application_data,recompute,predict,explain_prediction
from fastapi import HTTPException
router=APIRouter(prefix="/predict",tags=["predict"])
@router.post("/",response_model=Output)

def predictions(payload:User_input):
    model, selected_features, df = get_models()
    input=get_application_data(df,payload.Id)
    if input.empty:
        raise HTTPException(status_code=404, detail="Applicant ID not found")
    input["AMT_INCOME_TOTAL"] = payload.Income
    input["AMT_CREDIT"] = payload.Credit
    input["AMT_ANNUITY"] = payload.Annuity
    input["AMT_GOODS_PRICE"] = payload.Good_price
    input["CNT_FAM_MEMBERS"] = payload.Family_members
    input["DAYS_EMPLOYED"] = payload.Days_employed
    input = recompute(input)
    prob = predict(model, input, selected_features)
    if prob<0.30:
        risk_band="Low"
    elif prob<0.60:
        risk_band="Medium"
    else:
        risk_band="High"
    shap_values, base_value = explain_prediction(model,input,selected_features)
    paired = list(zip(selected_features, shap_values[0]))
    top_10 = sorted(paired, key=lambda x: abs(x[1]), reverse=True)[:10]
    shap_list = [ShapImpact(feature=f, impact=float(i)) for f, i in top_10]
    return Output(probability=prob, risk_band=risk_band, shap_values=shap_list)
