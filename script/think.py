#step 1-> LOOK at cols
import pandas as pd
base_path=r"C:\Users\PC\OneDrive\Desktop\work\home-credit-default-risk"

print("before installment \n")
inst=pd.read_csv(f"{base_path}\\installments_payments.csv")
print(inst.columns.tolist())
print(inst.shape)
print(inst.head)

print("\n after installment and before bureau \n")
bureau=pd.read_csv(f"{base_path}\\bureau.csv")
print(bureau.columns.tolist())
print(bureau.shape)
print(bureau.head)

print('\n after bureau,installment and before credit card balance\n')
cos=pd.read_csv(f"{base_path}\\credit_card_balance.csv")
print(cos.columns.tolist())
print(cos.shape)
print(cos.head)

print('\n after bureau, credit card balance installment payment and before pos cash balance\n ')
posh=pd.read_csv(f"{base_path}\\POS_CASH_balance.csv")
print(posh.columns.tolist())
print(posh.shape)
print(posh.head)

print('\n previous application \n')
prev=pd.read_csv(f"{base_path}\\previous_application.csv")
print(prev.columns.tolist())
print(prev.shape)
print(prev.head)



print('\n end \n')
