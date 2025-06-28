import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading

# ---------- 推理逻辑 ----------
from inference import load_model, generate
model, tokenizer, device = load_model()

# ---------- 通用函数 ----------
def async_run(cmd, output_box):
    def run():
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(process.stdout.readline, ""):
                output_box.insert(tk.END, line)
                output_box.see(tk.END)
            process.stdout.close()
        except Exception as e:
            messagebox.showerror("运行错误", str(e))
    threading.Thread(target=run).start()

# ---------- GUI 主界面 ----------
root = tk.Tk()
root.title("FluxProp 一体化平台")
root.geometry("800x600")

tab_control = ttk.Notebook(root)

# Tab1: 推理界面
infer_tab = ttk.Frame(tab_control)
tab_control.add(infer_tab, text='🧠 推理')

prompt_label = tk.Label(infer_tab, text="请输入提示词：")
prompt_label.pack()
prompt_input = tk.Entry(infer_tab, width=100)
prompt_input.pack(pady=5)
output_box = scrolledtext.ScrolledText(infer_tab, width=100, height=20, wrap=tk.WORD)
output_box.pack(pady=5)

def on_generate():
    prompt = prompt_input.get().strip()
    if not prompt:
        messagebox.showwarning("提示", "请输入提示词")
        return
    output = generate(model, tokenizer, prompt, device)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, output)

generate_button = tk.Button(infer_tab, text="生成", command=on_generate)
generate_button.pack(pady=10)

# Tab2: 训练界面
train_tab = ttk.Frame(tab_control)
tab_control.add(train_tab, text='🏋️‍♀️ 训练')

tk.Label(train_tab, text="训练层编号:").pack()
layer_entry = tk.Entry(train_tab)
layer_entry.pack(pady=5)

train_path = tk.StringVar()
def choose_train_file():
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if path:
        train_path.set(path)

tk.Button(train_tab, text="选择 train.txt", command=choose_train_file).pack()
tk.Label(train_tab, textvariable=train_path, wraplength=600).pack(pady=5)

train_output_box = scrolledtext.ScrolledText(train_tab, width=100, height=20)
train_output_box.pack(pady=10)

def start_training():
    layer = layer_entry.get().strip()
    if not layer.isdigit():
        messagebox.showerror("错误", "层编号必须是整数")
        return
    cmd = ["python", "train_layer.py", "--layer", layer]
    async_run(cmd, train_output_box)

tk.Button(train_tab, text="开始训练", command=start_training).pack(pady=10)

tab_control.pack(expand=1, fill="both")
root.mainloop()
