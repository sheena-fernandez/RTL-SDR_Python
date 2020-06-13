# RTL-SDR_Python
Python implementation for RTL-SDR FM application

<br/>

### A. Install RTL-SDR Driver
* #### Windows
    Install through [Zadig](https://zadig.akeo.ie)

* #### macOS
    ```
    brew install librtlsdr
    ```

<br/>

### B. Setup Python environment
#### 1. Python Installation
* #### Windows
    Download Python from [Python Installers](https://www.python.org/downloads/)

* #### macOS
    ```
    brew install python3
    ```

#### 2. Verify Python Installation
```
python3 --version
```

<br/>

### C. Install Project environment using Pipenv

```
pip install pipenv
pipenv install
pipenv shell
```
> Note: You can use pipenv run python main.py to use installed packages or pipenv shell to spawn a new shell which can run all available commands.

<br/>


### D. Installing PyAudio
PyAudio may not be automatically installed through pip. Execute the following to install PyAudio.

* #### Windows
1. Download the appropriate WHL file from [Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
2. Go to the installation folder and run
    ```
    pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
    ```

* #### MacOS
    ```
    brew install portaudio
    pip install pyaudio
    ```

### E. Usage: 
```
fm_sample.py
```
```
fm_stream.py <station number in MHz>
```

<br />

### F. Developing Python in VSCode Guide
[Getting Started with Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)