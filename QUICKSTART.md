# Quick Start Guide - Facial Recognition System

## Step 1: Install Python

1. Download Python 3.8 or higher from https://www.python.org/downloads/
2. During installation, **CHECK** "Add Python to PATH"
3. Complete the installation

## Step 2: Verify Python Installation

Open a terminal and run:

```bash
python --version
```

You should see something like: `Python 3.8.x` or higher

## Step 3: Install Visual Studio Build Tools (Windows Only)

**This is required for dlib to compile:**

1. Go to: https://visualstudio.microsoft.com/downloads/
2. Download "Build Tools for Visual Studio 2022"
3. Run the installer
4. Select "Desktop development with C++"
5. Click Install
6. Restart your computer after installation

## Step 4: Install Python Dependencies

Open a terminal in the project folder and run:

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies (this may take 5-10 minutes)
pip install -r requirements.txt
```

**Note**: Installing `dlib` will take the longest time as it compiles from source.

## Step 5: Run the Application

### Option A: Using Python directly

```bash
python main.py
```

### Option B: Using IntelliJ IDEA

1. Open IntelliJ IDEA
2. Install Python plugin: `File > Settings > Plugins > Search "Python" > Install`
3. Restart IntelliJ IDEA
4. Configure Python SDK: `File > Project Structure > SDKs > + > Add Python SDK`
5. Right-click `main.py` > Run 'main'

## Step 6: Use the Application

1. Click "▶ Start Camera"
2. Enter a name in the "Name" field
3. Click "Register New Face"
4. Look at the camera and move your head slightly
5. Wait for 10 samples to be captured
6. Registration complete! The system will now recognize you

## Troubleshooting

### Error: "No module named 'face_recognition'"

Run:
```bash
pip install face_recognition
```

### Error: "Failed to access camera"

- Close other apps using the camera (Zoom, Teams, etc.)
- Check camera permissions in Windows Settings
- Make sure camera is connected

### Error: "dlib installation failed"

- Make sure you installed Visual Studio Build Tools
- Try using a pre-built wheel:
  ```bash
  pip install cmake
  pip install dlib
  ```

### Application is slow

Edit `main.py` and reduce camera resolution:

Change line 29 from:
```python
self.camera = CameraHandler(width=640, height=480)
```

To:
```python
self.camera = CameraHandler(width=320, height=240)
```

## Need More Help?

See the full `README.md` for detailed documentation and troubleshooting.

---

**Enjoy your facial recognition system! 🎉**
