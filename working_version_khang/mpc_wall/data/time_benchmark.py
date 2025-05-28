import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_mpc_compute_time_logs(
    filename1="mpc_compute_time_log_RK4.txt",
    filename2="mpc_compute_time_log_collocation.txt" 
):
    """
    Loads MPC compute time logs from two text files and plots them for comparison.

    Args:
        filename1 (str): Path to the first log file (e.g., from direct shooting).
        filename2 (str): Path to the second log file (e.g., from collocation).
    """
    plt.figure(figsize=(12, 6))

    # --- Load and plot data from the first file ---
    try:
        data1 = np.loadtxt(filename1, skiprows=1) # skiprows=1 to skip the header
        plt.plot(data1 * 1000, label=f"RK4", alpha=0.8)
        print(f"Successfully loaded data from '{filename1}'")
        print(f"  Average compute time for {filename1}: {np.mean(data1)*1000:.3f} ms")
        print(f"  Max compute time for {filename1}: {np.max(data1)*1000:.3f} ms")
    except FileNotFoundError:
        print(f"Warning: '{filename1}' not found. Skipping this file.")
    except Exception as e:
        print(f"Error loading or plotting '{filename1}': {e}")

    # --- Load and plot data from the second file ---
    try:
        data2 = np.loadtxt(filename2, skiprows=1) # skiprows=1 to skip the header
        plt.plot(data2 * 1000, label=f"Collocation", alpha=0.8)
        print(f"Successfully loaded data from '{filename2}'")
        print(f"  Average compute time for {filename2}: {np.mean(data2)*1000:.3f} ms")
        print(f"  Max compute time for {filename2}: {np.max(data2)*1000:.3f} ms")
    except FileNotFoundError:
        print(f"Warning: '{filename2}' not found. Skipping this file.")
    except Exception as e:
        print(f"Error loading or plotting '{filename2}': {e}")

    # --- Plotting enhancements ---
    plt.xlabel("MPC Iteration")
    plt.ylabel("Compute Time (ms)")
    plt.title("MPC Compute Time per Iteration: RK4 vs. Collocation")
    plt.grid(True)
    plt.ylim(0, 150)  # Set y-axis limit to start from 0
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

# --- Call the function to plot your data ---
load_and_plot_mpc_compute_time_logs(
    filename1="mpc_compute_time_log_RK4.txt",
    filename2="mpc_compute_time_log_collocation.txt" 
)