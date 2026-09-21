from .cli import main

# The guard matters on Windows: multiprocessing workers re-import this module when they start.
if __name__ == "__main__":
    main()
