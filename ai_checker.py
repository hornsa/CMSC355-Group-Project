import tkinter as tk
from tkinter import ttk, messagebox
import joblib

#list of drugs
DRUG_LIST = [
    "aspirin", "warfarin", "ibuprofen", "alcohol", "acetaminophen", "vitaminC",
    "amoxicillin", "ginkgo", "vitaminK", "insulin", "grapefruit", "clopidogrel",
    "omeprazole", "digoxin", "verapamil", "lisinopril", "potassium", "sildenafil",
    "nitrates", "codeine", "diazepam", "alprazolam", "ketoconazole", "levothyroxine",
    "calcium", "ciprofloxacin", "antacids", "methotrexate", "NSAIDs", "tramadol",
    "SSRIs", "phenytoin", "folic acid", "fluoxetine", "sertraline", "doxycycline",
    "dairy", "nitroglycerin", "spironolactone", "beta-blockers", "hydrocodone",
    "clavulanate", "paracetamol", "caffeine", "metformin", "glipizide",
    "hydrochlorothiazide", "gabapentin", "oxycodone", "naproxen", "diclofenac",
    "metronidazole", "iron", "amlodipine", "simvastatin", "prednisone",
    "azithromycin", "cetirizine", "fluconazole", "loratadine", "montelukast",
    "metoprolol", "pantoprazole", "venlafaxine", "zolpidem", "alendronate",
    "levetiracetam", "allopurinol", "propranolol", "clonidine", "rosuvastatin",
    "pioglitazone", "rivaroxaban", "dabigatran", "esomeprazole", "lansoprazole",
    "famotidine", "itraconazole", "ondansetron", "lamotrigine", "topiramate",
    "eszopiclone", "vilazodone", "bupropion", "duloxetine", "lithium",
    "buprenorphine", "naloxone", "methadone"
]

#allow the model to be loaded
try:
    model = joblib.load("drug_interaction_model.pkl")
except FileNotFoundError:
    raise FileNotFoundError("Make sure 'drug_interaction_model.pkl' is in this folder.")

#the level of confidence required to predict an interaction
#if the confidence is below this the user will be asked to verify the interaction
threshold = 0.6
history_items = []
#function to predict interaction between two drugs
def predict_interaction(a, b):
    pair = f"{a.lower()} {b.lower()}"
    vec = model.named_steps['vectorizer'].transform([pair])
    if vec.nnz == 0:
        return "Unfamiliar combination. Please verify."
    pred = model.predict([pair])[0]
    conf = max(model.predict_proba([pair])[0])
    if pred == 0:
        return "No known interaction."
    if conf < threshold:
        return "Unfamiliar combination. Please verify."
    return "Potential interaction!"
#function to run the GUI
def run_gui():
    global threshold

    root = tk.Tk()
    root.title("AI Drug Interaction Checker")
    root.state('zoomed')
    root.configure(background='#f0f0f0')

#style configuration
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('Header.TLabel', background='#2a9d8f', foreground='white',
                    font=('Times New Roman', 20, 'bold'), padding=10)
    ACCENT, ACCENT_HOVER = '#0078D7', '#005A9E'
    style.configure('Accent.TButton', foreground='white', background=ACCENT,
                    font=('Times New Roman', 11, 'bold'), padding=8)
    style.map('Accent.TButton',
              background=[('active', ACCENT_HOVER), ('!disabled', ACCENT)],
              foreground=[('!disabled', 'white')])
    style.configure('Field.TLabel', background='#f0f0f0',
                    font=('Times New Roman', 12, 'bold'), padding=5)
    style.configure('TLabelframe.Label', font=('Times New Roman', 14, 'bold'))
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0', foreground='#000000')
    style.configure('TEntry', fieldbackground='white', foreground='#000000')
    style.configure('TCombobox', fieldbackground='white', foreground='#000000')

#header
    ttk.Label(root, text="AI Drug Checker", style='Header.TLabel').pack(fill='x')

#input fields
    inp = ttk.Frame(root, padding=20)
    inp.pack(fill='x')
    inp.columnconfigure(1, weight=1)

    ttk.Label(inp, text="First drug:", style='Field.TLabel').grid(row=0, column=0, sticky='e')
    combo1 = ttk.Combobox(inp, values=DRUG_LIST)
    combo1.grid(row=0, column=1, sticky='ew', padx=(5,0))

    ttk.Label(inp, text="Second drug:", style='Field.TLabel').grid(row=1, column=0, sticky='e', pady=(10,0))
    combo2 = ttk.Combobox(inp, values=DRUG_LIST)
    combo2.grid(row=1, column=1, sticky='ew', padx=(5,0), pady=(10,0))

    result_label = ttk.Label(inp, text="", anchor='center', wraplength=600,
                             font=('Times New Roman', 12, 'bold'))
    result_label.grid(row=3, column=0, columnspan=2, pady=(20,0))
    conf_var = tk.DoubleVar(value=threshold)

#buttons
    btns = ttk.Frame(root)
    btns.pack(fill='x', pady=(0,10))
    ttk.Button(btns, text="Evaluate Interaction", style='Accent.TButton',
               command=lambda: on_check(combo1, combo2, result_label, history_list, conf_var, search_var)).pack(side='left', padx=10)
    ttk.Button(btns, text="Reset Inputs", style='Accent.TButton',
               command=lambda: on_clear(combo1, combo2, result_label)).pack(side='left', padx=10)

#history
    hist_frame = ttk.LabelFrame(root, text="History", padding=10)
    hist_frame.pack(fill='x', expand=False, padx=20, pady=(0,10))

    search_var = tk.StringVar()
    search_frame = ttk.Frame(hist_frame)
    search_frame.pack(fill='x')
    ttk.Label(search_frame, text="Search:", style='Field.TLabel').pack(side='left')
    search_entry = ttk.Entry(search_frame, textvariable=search_var)
    search_entry.pack(side='left', fill='x', expand=True, padx=(5,0))
    search_entry.bind("<KeyRelease>", lambda e: update_history(history_list, search_var))

    list_frame = ttk.Frame(hist_frame)
    list_frame.pack(fill='x', pady=(10,0))
    history_list = tk.Listbox(list_frame, font=('Times New Roman', 14), height=6)
    history_list.pack(side='left', fill='x', expand=True)
    scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=history_list.yview)
    scrollbar.pack(side='right', fill='y')
    history_list.config(yscrollcommand=scrollbar.set)

#settings
    cfg = ttk.LabelFrame(root, text="Settings", padding=10)
    cfg.pack(fill='x', padx=20, pady=(0,20))
    cfg.columnconfigure(1, weight=1)
#slider for confidence threshold
    ttk.Label(cfg, text="Confidence threshold:", style='Field.TLabel')\
       .grid(row=0, column=0, sticky='w')
    conf_scale = ttk.Scale(cfg, from_=0.0, to=1.0, orient='horizontal', variable=conf_var)
    conf_scale.grid(row=0, column=1, sticky='ew', padx=(10,0))
    pct_label = ttk.Label(cfg, text=f"{int(conf_var.get()*100)}%", style='Field.TLabel')
    pct_label.grid(row=0, column=2, padx=(10,0))
    conf_scale.bind("<Motion>", lambda e: pct_label.config(text=f"{int(conf_var.get()*100)}%"))
    conf_scale.bind("<ButtonRelease-1>", lambda e: pct_label.config(text=f"{int(conf_var.get()*100)}%"))

    btn_frame = ttk.Frame(cfg)
    btn_frame.grid(row=1, column=0, columnspan=3, pady=(10,0))
    for i in range(1, 11):
        val = i * 10
        ttk.Button(btn_frame, text=f"{val}%", style='Accent.TButton',
                   command=lambda v=val: (conf_var.set(v/100), pct_label.config(text=f"{v}%")))\
            .pack(side='left', padx=2)
#apply button
    apply_btn = ttk.Button(cfg, text="Apply", style='Accent.TButton',
                           command=lambda: apply_settings(conf_var))
    apply_btn.grid(row=2, column=0, columnspan=3, pady=(10,0))

#quit button
    bottom_frame = ttk.Frame(root)
    bottom_frame.pack(side='bottom', fill='x', pady=20)
    ttk.Button(bottom_frame, text="Quit", style='Accent.TButton',
               command=lambda: confirm_quit(root)).pack()

    root.mainloop()
#confirm quit message
def confirm_quit(root):
    if messagebox.askyesno("Confirm Exit", "Are you sure you want to quit?"):
        root.destroy()
#function to handle interaction check
def on_check(c1, c2, result_label, listbox, conf_var, search_var):
    global threshold
    threshold = conf_var.get()

    d1, d2 = c1.get().strip(), c2.get().strip()
    if not d1 or not d2:
        result_label.config(text="Please select both drugs.")
        return

    res = predict_interaction(d1, d2)
    result_label.config(text=res)
    history_items.append(f"{d1} + {d2} → {res}")
    update_history(listbox, search_var)
#clear inputs and result
def on_clear(c1, c2, result_label):
    c1.set('')
    c2.set('')
    result_label.config(text='')
#update history
def update_history(listbox, sv):
    term = sv.get().lower() if isinstance(sv, tk.StringVar) else sv.lower()
    listbox.delete(0, tk.END)
    for item in history_items:
        if term in item.lower():
            listbox.insert(tk.END, item)
#apply settings
def apply_settings(var):
    global threshold
    threshold = var.get()

if __name__ == "__main__":
    run_gui()
