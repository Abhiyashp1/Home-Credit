import os
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")

model = pickle.load(open(os.path.join(BACKEND_DIR, "model.pkl"), 'rb'))

xgb.plot_importance(model, max_num_features=15)
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "feature_importance.png"))
plt.show()
print("Saved feature_importance.png")
