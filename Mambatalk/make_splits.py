import pandas as pd
import numpy as np

df = pd.read_csv("train_test_split_original.csv")
np.random.seed(42)

# Locuteurs que tu utilises
speakers = [2, 4, 9]

train_mask = (df["type"] == "train") & (
    df["id"].str.split("_").str[0].astype(int).isin(speakers)
)
train_df = df[train_mask]
print(f"Total séquences train (speakers {speakers}) : {len(train_df)}")

# Vérification par locuteur
for sp in speakers:
    sp_mask = train_df["id"].str.split("_").str[0].astype(int) == sp
    print(f"  Speaker {sp} : {sp_mask.sum()} séquences")

for ratio, name in [(0.5, "50"), (0.25, "25")]:
    df_split = df.copy()
    selected_indices = []
    
    # ✅ Stratification : on réduit chaque locuteur séparément
    for sp in speakers:
        sp_mask = (df["type"] == "train") & (
            df["id"].str.split("_").str[0].astype(int) == sp
        )
        sp_indices = df[sp_mask].index.tolist()
        n_keep = max(1, int(len(sp_indices) * ratio))
        kept = np.random.choice(sp_indices, size=n_keep, replace=False)
        selected_indices.extend(kept)
        print(f"  [{name}%] Speaker {sp} : {n_keep}/{len(sp_indices)} séquences gardées")
    
    # Les train non sélectionnés → additional
    all_train_indices = train_df.index.tolist()
    excluded = set(all_train_indices) - set(selected_indices)
    df_split.loc[list(excluded), "type"] = "additional"
    
    df_split.to_csv(f"train_test_split_{name}.csv", index=False)
    print(f"→ Sauvegardé : train_test_split_{name}.csv\n")