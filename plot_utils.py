import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def smooth(data, window_size=20):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_loss_and_memory(loss_log, mem_log, save_path=None):
    smoothed_loss = smooth(loss_log)
    steps = list(range(len(smoothed_loss)))

    plt.figure(figsize=(12, 6))

    # Loss 图
    plt.subplot(2, 1, 1)
    plt.plot(steps, smoothed_loss, label='Smoothed Loss')
    plt.ylabel("Loss")
    plt.title("Training Loss and GPU Memory Usage")
    plt.grid(True)
    plt.legend()

    # Memory 图
    plt.subplot(2, 1, 2)
    plt.plot(mem_log, label='GPU Memory (MB)')
    plt.axhline(max(mem_log), color="red", linestyle="--", label="Max Mem")
    plt.text(5, max(mem_log) + 0.5, f"Max: {max(mem_log):.1f} MB", color="red")
    plt.ylabel("Memory (MB)")
    plt.xlabel("Step")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

