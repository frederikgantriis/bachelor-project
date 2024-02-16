### Initial setup

1. Create a virtual environment from the root of the project folder

    ```sh
    python3 -m venv env
    ```

2. Activate the environment (make sure it is activated every time you work on the project)

    ```sh
    source env/bin/activate
    ```

    Windows:

    ```sh
    .\env\Scripts\Activate.ps1
    ```

    Typing `deactivate` in the shell will exit the environment

3. Update pip

    ```sh
    pip install --upgrade pip
    ```

1. Install requirements from the list

    ```sh
    pip install -r requirements.txt
    ```

### Saving and Loading requirements

This is only needed if packages needs to be synced with e.g. collaborators or your future self

1. Save a formatted list of packages in a `requirements.txt` file

    ```sh
    pip freeze > requirements.txt
    ```

1. Install requirements from the list

    ```sh
    pip install -r requirements.txt
    ```

### Test project

Here are the commands for testing the project

1. For running basic test
    ```sh
    pytest
    ```
1. For getting the test coverage
    ```sh 
    pytest --cov=. --cov-fail-under=90
    ```



