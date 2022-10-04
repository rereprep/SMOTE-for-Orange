# Get libraries
import Orange
import numpy as np
from Orange.data import Domain, Table
from imblearn.over_sampling import SMOTE

# How Orange passes data to widget
df = in_data.copy()

# set variables for SMOTE
sm = SMOTE(random_state=42)

# get table of data (X) and class variables (y)
X, y = df.X, df.Y

# resample data and classes
X_res, y_res = sm.fit_resample(X, y)

# Get the target and feature variables
d = Domain(df.domain.attributes, df.domain.class_vars)

# Create a new Orange Table object with the appropriate headers
# This is how Orange passes the data on to the next widget
out_data = Orange.data.Table(d, X_res, y_res)
