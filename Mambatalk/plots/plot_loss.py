import json
import matplotlib.pyplot as plt

with open("outputs/audio2pose/custom/0524_171228_mambatalk/l2_loss_history.json", "r") as f:
    data = json.load(f)

epochs      = [d["epoch"] for d in data]
latent_loss = [d["latent_loss"] for d in data]
cls_loss    = [d["cls_loss"] for d in data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, latent_loss, color='blue', linewidth=2)
ax1.set_xlabel("Époques")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Latent Loss (MSE)")
ax1.grid(True)

ax2.plot(epochs, cls_loss, color='orange', linewidth=2)
ax2.set_xlabel("Époques")
ax2.set_ylabel("Cross-Entropy Loss")
ax2.set_title("Classification Loss (CLS)")
ax2.grid(True)

plt.suptitle("Évolution des losses d'entraînement - Run 25%", fontsize=13)
plt.tight_layout()
plt.savefig("plots/train_losses_25_3.png", dpi=150)
plt.show()
print("Sauvegardé : train_losses_25_3.png")