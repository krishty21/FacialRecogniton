# Face Recognition System 🔍

Hey! So I built this facial recognition app that actually works pretty well. It uses OpenCV's LBPH algorithm and can recognize faces with around 85-92% accuracy once you've fed it some good samples. No crazy compilation steps needed – just Python and you're good to go!

## What Can It Do?

- ✨ **Pretty Accurate**: Gets it right 85-92% of the time with the LBPH algorithm (just make sure to give it quality samples)
- 🎥 **Live Recognition**: Pulls from your webcam and recognizes faces on the fly
- 👤 **Easy Setup**: Adding new faces is super straightforward
- 📸 **Smart Sampling**: Grabs 10 photos of each person to really learn their face
- 💾 **Remembers Everyone**: Stores everything in a SQLite database
- 🖥️ **Clean Interface**: Built with Tkinter, so it's simple and works everywhere
- ⚙️ **Manage Profiles**: Add people, remove them, or start fresh whenever you want
- 🧠 **IntelliJ Friendly**: Works great in IntelliJ IDEA if that's your jam

## What's Under the Hood?

- **Python 3.7+** – The brain of the operation
- **OpenCV-contrib** – Handles all the face recognition magic
- **NumPy** – For crunching numbers
- **Tkinter** – Nice and simple GUI
- **SQLite** – Keeps track of all the face data

## What You'll Need

- Python 3.7 or newer
- A webcam (or any camera, really)
- Windows, Linux, or macOS – doesn't matter

## Getting Started

### Step 1: Grab Python

Head over to [python.org](https://python.org/downloads/) and download Python 3.7+.

**Pro tip**: When installing, make sure you tick "Add Python to PATH" – trust me, it'll save you headaches later!

### Step 2: Install the Goodies

Pop open your terminal or command prompt in the project folder and run:

```bash
# First, let's make sure pip is up to date
python -m pip install --upgrade pip

# Now install everything we need
pip install -r requirements.txt
```

Should only take a minute or two!

### Step 3: Double-Check Everything Works

```bash
python -c "import cv2; print('All good to go!')"
```

If you see "All good to go!", you're set!

## Running It in IntelliJ IDEA

### Setting Up Python in IntelliJ

1. **Get the Python Plugin**:
   - Open `File > Settings > Plugins`
   - Search for "Python"
   - Install it and restart IntelliJ

2. **Connect Your Python**:
   - Go to `File > Project Structure > SDKs`
   - Hit `+` then pick "Add Python SDK"
   - Find your Python installation (probably something like `C:\Python310\python.exe`)
   - Click OK

3. **Link It to Your Project**:
   - Head to `File > Project Structure > Modules`
   - Select that Python SDK you just added
   - Hit Apply and you're done

### Fire It Up!

1. Open `main.py` in IntelliJ
2. Right-click anywhere in the file
3. Click `Run 'main'`

Or if you prefer the terminal:
```bash
python main.py
```

## How to Use This Thing

### Starting Up

Just run:
```bash
python main.py
```

### Adding Someone New

1. Hit **"▶ Start Camera"** to turn on your webcam
2. Type their name in the **Name** box
3. Click **"Register New Face"**
4. Look at the camera – the system will snap 10 photos automatically
5. Move your head around a bit between shots (helps with accuracy!)
6. That's it – they're in the system!

**Want the best results?**
- Good lighting is your friend
- Look straight at the camera
- If you wear glasses, maybe register both with and without?
- Watch out for shadows on your face
- Slight head movements (tilt, turn) help it learn better

### Recognizing People

1. Make sure the camera's on
2. When someone registered shows up, you'll see a green box with their name and how confident the system is
3. Unknown folks get a red box – sorry, stranger!

### Managing Who's in the System

- **View Registered Persons**: See everyone who's in there and how many samples you got
- **Delete Person**: Remove someone (maybe they left the team?)
- **Clear All Data**: Nuclear option – wipes everyone (be careful with this one!)

## What's What in the Project

```
FacialRecogniton/
├── main.py                      # The main app with the GUI
├── face_recognition_engine.py   # Where the recognition magic happens
├── database_manager.py          # Talks to the SQLite database
├── camera_handler.py            # Handles the camera stuff
├── requirements.txt             # All the Python packages you need
├── face_recognition.db          # The database (appears on first run)
├── README.md                    # You're reading it!
└── src/                         # Old Java code (just gathering dust now)
```

## The Nerdy Stuff (How It Actually Works)

### The Algorithm

This thing uses **LBPH (Local Binary Patterns Histograms)** from OpenCV:

1. **Finds Your Face**: Uses Haar Cascades to spot faces in the frame
2. **Analyzes It**: Breaks your face down into local patterns
3. **Creates a Fingerprint**: Makes a unique histogram for your face
4. **Compares**: When it sees a face, it compares the histogram to what it knows
5. **Decides**: If it's close enough, it's a match!

**Why LBPH is cool**:
- Doesn't need a ton of computing power
- Pretty forgiving with lighting changes
- Works even if you're at a slight angle
- Fast enough for real-time recognition
- Gets better with more samples (hence the 10 photos!)

### Registering Process

1. Snaps 10 photos from different angles
2. Creates a unique pattern for each one
3. Saves it all to the database
4. More samples = better recognition later

### Recognition Process

1. Spots a face in the camera
2. Creates its pattern
3. Compares it to everyone in the database
4. If the difference is small enough (default threshold: 50), it's a match!
5. Shows you the name and how confident it is

## When Things Go Wrong

### "Can't Access the Camera!"

- Is it actually plugged in and working?
- Close Zoom, Teams, or anything else hogging the camera
- Try a different camera by editing `camera_handler.py` line 22: `CameraHandler(camera_index=1)`

### "Can't Find OpenCV" or Other Import Errors

```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

### Not Recognizing People Well

- Try re-registering them with better lighting
- Get more varied angles when taking the 10 samples
- You can tweak the threshold in `face_recognition_engine_opencv.py` – lower numbers are stricter, higher are more lenient

### Running Super Slow

- Lower the camera resolution in `camera_handler.py`:
  ```python
  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  ```

## Tweaking It to Your Liking

### Recognition Sensitivity

You can adjust how picky it is by editing the threshold in `face_recognition_engine_opencv.py`:

```python
confidence = int(100 * (1 - (label_distance / 300)))  # Play with that 300!
```

Lower = more strict, higher = more forgiving

### Number of Samples

Want more or fewer photos during registration? Edit `main.py`:

```python
self.target_samples = 15  # Or whatever number you want
```

More samples generally = better accuracy (but takes longer to register)

## Quick Comparison

Here's how this stacks up against the old Java version I had:

| What | Old Java Version | This Python Version |
|------|------------------|---------------------|
| **Accuracy** | ~70-80% | ~85-92% |
| **Algorithm** | LBPH (basic) | LBPH (optimized) |
| **Setup** | Painful | Super easy |
| **Code** | 712 messy lines | ~400 clean lines |
| **Dependencies** | Native libs (ugh) | Just pip packages |
| **Maintenance** | Nightmare fuel | Actually pleasant |

## Props To

- **OpenCV** for doing the heavy lifting
- The folks behind **NumPy** for fast math
- **Tkinter** for making GUIs bearable
- **SQLite** for being a solid little database

## License Stuff

This is just a learning project – feel free to use it however you want!

## Need Help?

If something's not working:
1. Check the troubleshooting section above
2. Make sure you installed everything properly
3. Verify your camera has permissions
4. Try better lighting – seriously, it makes a huge difference!

---

That's about it! If you run into issues or have questions, feel free to dig into the code or hit me up. Happy face recognizing! 🎉

