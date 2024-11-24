

###  Create a Virtual Environment

#### For macOS/Linux:

```bash
python3 -m venv venv
```

#### For Windows:

```bash
python -m venv venv
```

###  Activate the Virtual Environment

#### For macOS/Linux:

```bash
source venv/bin/activate
```

#### For Windows (Command Prompt):

```bash
.\venv\Scripts\activate
```

#### For Windows (PowerShell):

```bash
env\Scripts\Activate.ps1
```

###  Install Dependencies

Once the virtual environment is activated, install the dependencies listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

###  Deactivate the Environment

When you're done working, you can deactivate the virtual environment by running:

```bash
deactivate
```

###  Reactivate the Virtual Environment

Any time you want to work on the project again, you'll need to reactivate the virtual environment. Use the following commands based on your operating system.

#### macOS/Linux:

```bash
source venv/bin/activate
```

#### Windows:

```bash
.\venv\Scripts\activate
```

If you're adding new packages, make sure to update the `requirements.txt` file by running:

```bash
pip freeze > requirements.txt
```



### To Set Interpreter of Venv (For VSCode)


-Press ```Ctrl+Shift+P on windows or Cmd+Shift+P``` on macOS to open the Command Palette.

-Search for "Python: Select Interpreter".

-Choose the virtual environment.


