#!/usr/bin/env python3

import subprocess


def main():
    try:
        print("Attempting to push changes to Git...")
        print("If it fails, it will retry until it succeeds.")
        result = subprocess.run(
            ["git", "push"],
            capture_output=True,  # Capture both stdout and stderr
            text=True,            # Return output as string (not bytes)
            check=True            # Raise exception if command fails
        )

        print(f'{result.stdout=}')
        print(f'{result.stderr=}')
        print(f'{result.returncode=}')
        print("If you see this message, the push command was executed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode=}")
        print(f"Error output: {e.stderr=}")
        # Try again
        main()


if __name__ == '__main__':
    main()
