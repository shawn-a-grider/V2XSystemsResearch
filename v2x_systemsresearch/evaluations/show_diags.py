from training_diagnostics import TrainingDiagnostics

diagnostics = TrainingDiagnostics('/Users/shawngrider/av-modcollapse/logs/progress.csv')

diagnostics.plot_rew()
diagnostics.plot_kl()
diagnostics.plot_entropy_loss()
diagnostics.plot_explained_variance()