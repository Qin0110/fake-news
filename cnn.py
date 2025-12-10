from preprocessing import Preprocessing
import numpy as np

preprocessor = Preprocessing()

X_train, X_validate, X_test, y_train, y_validate, y_test = preprocessor.split_data(test_size=0.2)

print(f"Training set size: {len(X_train)}"
      f"\nValidation set size: {len(X_validate)}"
      f"\nTest set size: {len(X_test)}")

print(f"Training labels distribution: {np.bincount(y_train)}"
      f"\nValidation labels distribution: {np.bincount(y_validate)}"
        f"\nTest labels distribution: {np.bincount(y_test)}")
