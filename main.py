"""
Facial Recognition System - Main Application
Modern facial recognition with deep learning
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from typing import Optional

from database_manager import DatabaseManager
from face_recognition_engine_opencv import FaceRecognitionEngine
from camera_handler import CameraHandler


class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition System v3.0")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.face_engine = FaceRecognitionEngine(self.db_manager, tolerance=80.0)
        self.camera = CameraHandler(width=640, height=480)
        
        # State variables
        self.camera_active = False
        self.registration_mode = False
        self.registration_name = ""
        self.registration_count = 0
        self.target_samples = 10
        self.last_capture_time = 0
        self.capture_delay = 0.5  # seconds
        
        # Create GUI
        self.setup_gui()
        
        # Update statistics
        self.update_statistics()
        
        # Start camera feed thread
        self.running = True
        self.camera_thread = None
        
        self.log("Facial Recognition System initialized")
        self.log(f"Loaded {self.face_engine.get_statistics()['total_encodings']} face encodings")
    
    def setup_gui(self):
        """Create the GUI interface"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # === TOP PANEL - Statistics ===
        stats_frame = ttk.LabelFrame(main_container, text="System Status", padding="10")
        stats_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(stats_frame, text="Status: Ready", font=("Arial", 11, "bold"))
        self.status_label.grid(row=0, column=0, padx=20)
        
        self.persons_label = ttk.Label(stats_frame, text="Registered: 0", font=("Arial", 11))
        self.persons_label.grid(row=0, column=1, padx=20)
        
        self.encodings_label = ttk.Label(stats_frame, text="Encodings: 0", font=("Arial", 11))
        self.encodings_label.grid(row=0, column=2, padx=20)
        
        self.recognition_label = ttk.Label(stats_frame, text="Recognition: N/A", 
                                          font=("Arial", 11, "bold"), foreground="green")
        self.recognition_label.grid(row=0, column=3, padx=20)
        
        # === LEFT PANEL - Camera Feed ===
        camera_frame = ttk.LabelFrame(main_container, text="Live Camera Feed", padding="5")
        camera_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.camera_label = tk.Label(camera_frame, bg="black", text="Camera feed will appear here",
                                     fg="white", font=("Arial", 14))
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # === RIGHT PANEL - Controls ===
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.rowconfigure(1, weight=1)
        
        # Control buttons
        controls_frame = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_btn = ttk.Button(controls_frame, text="▶ Start Camera", 
                                    command=self.toggle_camera, width=25)
        self.start_btn.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Separator(controls_frame, orient='horizontal').grid(row=1, column=0, 
                                                                columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Registration section
        ttk.Label(controls_frame, text="Name:", font=("Arial", 10)).grid(row=2, column=0, 
                                                                         sticky=tk.W, pady=5)
        self.name_entry = ttk.Entry(controls_frame, width=20, font=("Arial", 10))
        self.name_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        self.register_btn = ttk.Button(controls_frame, text="Register New Face", 
                                       command=self.start_registration, width=25, state=tk.DISABLED)
        self.register_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.progress = ttk.Progressbar(controls_frame, maximum=self.target_samples, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_label = ttk.Label(controls_frame, text="Ready", font=("Arial", 9))
        self.progress_label.grid(row=5, column=0, columnspan=2, pady=2)
        
        ttk.Separator(controls_frame, orient='horizontal').grid(row=6, column=0, 
                                                                columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Management buttons
        self.view_btn = ttk.Button(controls_frame, text="👤 View Registered Persons", 
                                   command=self.view_persons, width=25)
        self.view_btn.grid(row=7, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.delete_btn = ttk.Button(controls_frame, text="🗑 Delete Person", 
                                     command=self.delete_person, width=25)
        self.delete_btn.grid(row=8, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.clear_btn = ttk.Button(controls_frame, text="⚠ Clear All Data", 
                                    command=self.clear_all, width=25)
        self.clear_btn.grid(row=9, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Log area
        log_frame = ttk.LabelFrame(right_panel, text="System Log", padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=20, width=40, font=("Consolas", 9), 
                               wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scroll.set)
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.camera_active:
            if self.camera.start():
                self.camera_active = True
                self.start_btn.configure(text="⏹ Stop Camera")
                self.register_btn.configure(state=tk.NORMAL)
                self.status_label.configure(text="Status: Camera Active", foreground="green")
                self.log("Camera started successfully")
                
                # Start camera thread
                self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
                self.camera_thread.start()
            else:
                messagebox.showerror("Error", "Failed to access camera!\nPlease check if camera is connected.")
                self.log("ERROR: Failed to start camera")
        else:
            self.camera_active = False
            self.camera.stop()
            self.start_btn.configure(text="▶ Start Camera")
            self.register_btn.configure(state=tk.DISABLED)
            self.status_label.configure(text="Status: Camera Stopped", foreground="black")
            self.log("Camera stopped")
            
            # Clear camera display
            self.camera_label.configure(image='', text="Camera feed will appear here")
    
    def camera_loop(self):
        """Main camera processing loop"""
        while self.camera_active and self.running:
            success, frame = self.camera.read_frame()
            
            if not success:
                time.sleep(0.01)
                continue
            
            # Process frame based on mode
            if self.registration_mode:
                self.process_registration_frame(frame)
            else:
                self.process_recognition_frame(frame)
            
            time.sleep(0.01)
    
    def process_registration_frame(self, frame):
        """Process frame during registration"""
        # Detect faces
        face_locations = self.face_engine.detect_faces(frame, model="hog")
        
        # Convert to BGR for OpenCV drawing
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if len(face_locations) != 1:
            msg = "No face detected" if len(face_locations) == 0 else "Multiple faces detected"
            cv2.putText(display_frame, msg, (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Draw registration box
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 200, 255), 3)
            
            # Capture sample if enough time has passed
            current_time = time.time()
            if current_time - self.last_capture_time >= self.capture_delay:
                if self.registration_count < self.target_samples:
                    success, msg = self.face_engine.register_face(frame, self.registration_name)
                    
                    if success:
                        self.registration_count += 1
                        self.last_capture_time = current_time
                        
                        self.root.after(0, lambda: self.progress.configure(value=self.registration_count))
                        self.root.after(0, lambda: self.progress_label.configure(
                            text=f"Capturing: {self.registration_count}/{self.target_samples}"))
                        
                        if self.registration_count % 2 == 0:
                            self.log(f"Captured {self.registration_count}/{self.target_samples} samples")
                        
                        if self.registration_count >= self.target_samples:
                            self.complete_registration()
            
            # Show progress
            cv2.putText(display_frame, f"Capturing: {self.registration_count}/{self.target_samples}", 
                       (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        self.display_frame(display_frame)
    
    def process_recognition_frame(self, frame):
        """Process frame for face recognition"""
        # Recognize faces
        results = self.face_engine.recognize_faces(frame, model="hog")
        
        # Convert to BGR for OpenCV drawing
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for face_location, name, confidence in results:
            top, right, bottom, left = face_location
            
            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.1f}%)" if name != "Unknown" else "Unknown"
            
            # Background for text
            cv2.rectangle(display_frame, (left, top - 30), (right, top), color, -1)
            cv2.putText(display_frame, label, (left + 5, top - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update recognition label
            if name != "Unknown":
                self.root.after(0, lambda n=name, c=confidence: 
                    self.recognition_label.configure(text=f"Recognition: {n} ({c:.1f}%)"))
        
        if not results:
            self.root.after(0, lambda: self.recognition_label.configure(text="Recognition: N/A"))
        
        self.display_frame(display_frame)
    
    def display_frame(self, frame):
        """Display frame in GUI"""
        # Resize for display
        display_height = 480
        display_width = 640
        frame_resized = cv2.resize(frame, (display_width, display_height))
        
        # Convert to PhotoImage
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk, text="")
    
    def start_registration(self):
        """Start face registration process"""
        name = self.name_entry.get().strip()
        
        if not name:
            messagebox.showwarning("Invalid Name", "Please enter a name")
            return
        
        if not name.replace("_", "").replace("-", "").isalnum():
            messagebox.showwarning("Invalid Name", 
                                  "Name can only contain letters, numbers, hyphens, and underscores")
            return
        
        # Check if person exists
        if self.db_manager.person_exists(name):
            result = messagebox.askyesno("Person Exists", 
                                        f"'{name}' is already registered.\nAdd more samples?")
            if not result:
                return
        
        self.registration_name = name
        self.registration_count = 0
        self.registration_mode = True
        self.last_capture_time = 0
        
        self.register_btn.configure(state=tk.DISABLED)
        self.name_entry.configure(state=tk.DISABLED)
        self.progress.configure(value=0)
        self.progress_label.configure(text="Starting...")
        self.status_label.configure(text=f"Status: Registering {name}", foreground="orange")
        
        self.log(f"Starting registration for: {name}")
        self.log("Please look at the camera and move your head slightly")
    
    def complete_registration(self):
        """Complete the registration process"""
        self.registration_mode = False
        
        self.root.after(0, lambda: self.register_btn.configure(state=tk.NORMAL))
        self.root.after(0, lambda: self.name_entry.configure(state=tk.NORMAL))
        self.root.after(0, lambda: self.name_entry.delete(0, tk.END))
        self.root.after(0, lambda: self.progress_label.configure(text="Complete!"))
        self.root.after(0, lambda: self.status_label.configure(text="Status: Registration Complete", 
                                                               foreground="green"))
        
        self.log(f"Registration complete for {self.registration_name}")
        self.log(f"Captured {self.target_samples} face samples")
        
        self.update_statistics()
        
        messagebox.showinfo("Success", 
                           f"Successfully registered {self.registration_name}!\n"
                           f"Captured {self.target_samples} face samples.\n"
                           "Recognition is now active.")
    
    def view_persons(self):
        """Show list of registered persons"""
        persons = self.db_manager.get_all_persons()
        
        if not persons:
            messagebox.showinfo("No Persons", "No persons have been registered yet.")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Registered Persons")
        dialog.geometry("500x400")
        
        # Create treeview
        tree_frame = ttk.Frame(dialog, padding="10")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        tree = ttk.Treeview(tree_frame, columns=("Name", "Samples", "Date"), show="headings")
        tree.heading("Name", text="Name")
        tree.heading("Samples", text="Samples")
        tree.heading("Date", text="Registered Date")
        
        tree.column("Name", width=150)
        tree.column("Samples", width=100)
        tree.column("Date", width=200)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate data
        for name, count, date in persons:
            # Format date
            try:
                date_obj = datetime.fromisoformat(date)
                date_str = date_obj.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = date
            
            tree.insert("", tk.END, values=(name, count, date_str))
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def delete_person(self):
        """Delete a registered person"""
        persons = self.db_manager.get_all_persons()
        
        if not persons:
            messagebox.showinfo("No Persons", "No persons have been registered yet.")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Delete Person")
        dialog.geometry("300x400")
        
        ttk.Label(dialog, text="Select person to delete:", font=("Arial", 10, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(dialog, font=("Arial", 10))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for name, count, _ in persons:
            listbox.insert(tk.END, f"{name} ({count} samples)")
        
        def confirm_delete():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a person to delete")
                return
            
            name = persons[selection[0]][0]
            
            result = messagebox.askyesno("Confirm Delete", 
                                        f"Are you sure you want to delete '{name}'?\n"
                                        "This cannot be undone!")
            
            if result:
                self.face_engine.delete_person(name)
                self.log(f"Deleted person: {name}")
                self.update_statistics()
                dialog.destroy()
                messagebox.showinfo("Success", f"'{name}' has been deleted.")
        
        ttk.Button(dialog, text="Delete", command=confirm_delete).pack(pady=5)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)
    
    def clear_all(self):
        """Clear all registered persons"""
        result = messagebox.askyesno("Confirm Clear All", 
                                    "Delete ALL registered persons and data?\n\n"
                                    "This action CANNOT be undone!",
                                    icon=messagebox.WARNING)
        
        if result:
            self.face_engine.clear_all_faces()
            self.log("All data cleared")
            self.update_statistics()
            messagebox.showinfo("Success", "All data has been cleared.")
    
    def update_statistics(self):
        """Update statistics display"""
        stats = self.face_engine.get_statistics()
        self.persons_label.configure(text=f"Registered: {stats['total_persons']}")
        self.encodings_label.configure(text=f"Encodings: {stats['total_encodings']}")
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
        print(formatted.strip())
    
    def on_closing(self):
        """Handle window close event"""
        self.running = False
        self.camera_active = False
        self.camera.stop()
        self.db_manager.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
