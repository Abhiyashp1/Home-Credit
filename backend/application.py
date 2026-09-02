from fastapi import APIRouter,HTTPException,status
from model_store import get_models
from schemas import Application_detail
from inference import get_application_data
router=APIRouter(prefix="/applicants",tags=["applicants"])
@router.get("/")
def get_application():
    model, selected_features, df = get_models()
    return df["SK_ID_CURR"].tolist()
@router.get("/{sk_id}",response_model=Application_detail)
def get_input(sk_id:int):
    model, selected_features, df = get_models()
    input= get_application_data(df, sk_id)
    if input.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Id not found")
    return Application_detail(
        Income=float(input["AMT_INCOME_TOTAL"].values[0]),
        Credit=float(input["AMT_CREDIT"].values[0]),
        Annuity=float(input["AMT_ANNUITY"].values[0]),
        Good_price=float(input["AMT_GOODS_PRICE"].values[0]),
        Family_members=int(input["CNT_FAM_MEMBERS"].values[0]),
        Days_employed=float(input["DAYS_EMPLOYED"].values[0])
    )
