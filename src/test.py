import pandas as pd

results_df = pd.DataFrame({
    'Fine-Tuning Iteration': [],
    })


results_df.loc[0] = [
    ['hi', 'sup']
]

results_df.loc[1] = [
    ['hi', 'sup', 'hey', 'ho']
]

print(results_df)