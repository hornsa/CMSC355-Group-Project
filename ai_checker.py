import tkinter as tk
import joblib

# ── Model loading ──────────────────────────────────────────────────────────────
try:
    model = joblib.load("drug_interaction_model.pkl")
except FileNotFoundError:
    raise FileNotFoundError("Make sure 'drug_interaction_model.pkl' is in this folder.")

def predict_interaction(primary, secondary):
    """Return a result string for the given two drug names."""
    pair = f"{primary.lower()} {secondary.lower()}"
    pred = model.predict([pair])[0]
    probs = model.predict_proba([pair])[0]
    conf = max(probs)
    if conf < 0.6:
        return "Unfamiliar combination. Please verify with a professional."
    return "⚠️ Potential interaction!" if pred == 1 else "✅ No known interaction."


# ── GUI setup ─────────────────────────────────────────────────────────────────
def run_gui():
    root = tk.Tk()
    root.title("AI Drug Interaction Checker")
    root.geometry("350x200")

    # First‐drug entry
    tk.Label(root, text="First drug:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    entry1 = tk.Entry(root)
    entry1.grid(row=0, column=1, padx=5, pady=5)

    # Second‐drug entry
    tk.Label(root, text="Second drug:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    entry2 = tk.Entry(root)
    entry2.grid(row=1, column=1, padx=5, pady=5)

    # Result display
    result_label = tk.Label(root, text="", wraplength=300, font=("Arial", 10, "bold"))
    result_label.grid(row=3, column=0, columnspan=2, pady=10)

    # Button action
    def on_check():
        d1 = entry1.get().strip()
        d2 = entry2.get().strip()
        if not d1 or not d2:
            result_label.config(text="Please enter both drug names.")
        else:
            result_label.config(text=predict_interaction(d1, d2))

    tk.Button(root, text="Check Interaction", command=on_check)\
      .grid(row=2, column=0, columnspan=2, pady=10)

    root.mainloop()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_gui()
