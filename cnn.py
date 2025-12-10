from preprocessing import Preprocessing
import numpy as np
import joblib
import matplotlib.pyplot as plt

# preprocessor = Preprocessing()
# preprocessor.dump_variable(test_size=0.2)
# preprocessor.dump_variable(test_size=0.3)

X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.2_split_data.joblib')

real = np.sum(y_train == 0)
fake = np.sum(y_train == 1)

labels = ['Real', 'Fake']
values = [real, fake]
plt.bar(labels, values)
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.ylabel('Count')
plt.title('Distribution of Real vs Fake')
plt.show()

print(f"Training set size: {len(X_train)}"
      f"\nValidation set size: {len(X_validate)}"
      f"\nTest set size: {len(X_test)}")

print(f"Training labels distribution: {np.bincount(y_train)}"
      f"\nValidation labels distribution: {np.bincount(y_validate)}"
        f"\nTest labels distribution: {np.bincount(y_test)}")
