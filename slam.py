"""The launcher: the editor ONLY (the M5 overhaul's UI is the
application). The pre-editor loop was retired in M7.5 - its
capabilities live in the editor, backed by the SLAM step process (M7:
the pipeline runs in a spawned process, the UI thread never blocks)."""
import sys

if __name__ == "__main__":
    if "--legacy" in sys.argv:
        print("slam.py: --legacy was retired in M7.5 - the editor is "
              "the application (run python slam.py).")
        sys.exit(2)
    from slam.ui.editor import run_editor
    run_editor()
