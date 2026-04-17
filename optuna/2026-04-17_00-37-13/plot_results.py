import optuna
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)
import plotly.io as pio

# Check available studies
studies = optuna.get_all_study_summaries(storage="sqlite:///study.db")
print(f"Available studies: {[s.study_name for s in studies]}")

# Load the first study (assuming there's only one)
if studies:
    study_name = studies[0].study_name
    print(f"Loading study: {study_name}")
    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///study.db"
    )
else:
    raise ValueError("No studies found in database")

print(f"Loaded study with {len(study.trials)} trials")
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")

# Generate and save plots
print("\nGenerating plots...")

# Contour plot
fig = plot_contour(study)
pio.write_html(fig, "contour_plot.html")
print("Saved contour_plot.html")

# Optimization history
fig = plot_optimization_history(study)
pio.write_html(fig, "optimization_history.html")
print("Saved optimization_history.html")

# Parameter importances
fig = plot_param_importances(study)
pio.write_html(fig, "param_importances.html")
print("Saved param_importances.html")

# Parallel coordinate plot
fig = plot_parallel_coordinate(study)
pio.write_html(fig, "parallel_coordinate.html")
print("Saved parallel_coordinate.html")

# Slice plot
fig = plot_slice(study)
pio.write_html(fig, "slice_plot.html")
print("Saved slice_plot.html")

print("\nAll plots saved successfully!")
