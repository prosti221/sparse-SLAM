"""The launcher: the editor is the application (the SLAM pipeline
runs in its step worker)."""
if __name__ == "__main__":
    from slam.ui.editor import run_editor
    run_editor()
