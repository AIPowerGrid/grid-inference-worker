# Launcher for PyInstaller one-file exe. Do not run as __main__ from inside package,
# so relative imports in inference_worker.cli work when frozen.
if __name__ == "__main__":
    from inference_worker.cli import main
    main()
