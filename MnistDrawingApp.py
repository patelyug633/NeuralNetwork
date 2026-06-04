import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MNISTDrawingApp:
    def __init__(self, root, neural_network):
        self.root = root
        self.net = neural_network
        self.root.title("MNIST Neural Network Digit Recognizer")
        self.root.geometry("800x600")
        
        # Drawing parameters
        self.grid_size = 28
        self.cell_size = 15  # pixels per cell
        self.canvas_size = self.grid_size * self.cell_size
        
        # Create drawing canvas
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, 
                                 bg='black', highlightthickness=2, highlightbackground='gray')
        self.canvas.pack(pady=10)
        
        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        # Initialize drawing grid (28x28 array of 0s)
        self.drawing_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Create buttons frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        # Predict button
        self.predict_btn = tk.Button(button_frame, text="PREDICT", command=self.predict_digit,
                                      bg='green', fg='white', font=('Arial', 12, 'bold'),
                                      padx=20, pady=5)
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Reset button
        self.reset_btn = tk.Button(button_frame, text="RESET", command=self.reset_canvas,
                                    bg='red', fg='white', font=('Arial', 12, 'bold'),
                                    padx=20, pady=5)
        self.reset_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear button (same as reset but different name)
        self.clear_btn = tk.Button(button_frame, text="CLEAR", command=self.reset_canvas,
                                    bg='orange', fg='white', font=('Arial', 12, 'bold'),
                                    padx=20, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Prediction label
        self.prediction_label = tk.Label(root, text="Draw a digit and click PREDICT", 
                                          font=('Arial', 16, 'bold'), fg='blue')
        self.prediction_label.pack(pady=10)
        
        # Confidence frame
        confidence_frame = tk.Frame(root)
        confidence_frame.pack(pady=10)
        
        tk.Label(confidence_frame, text="Confidence Scores:", font=('Arial', 12)).pack()
        
        # Progress bars for each digit (0-9)
        self.confidence_bars = []
        self.confidence_labels = []
        
        bar_frame = tk.Frame(confidence_frame)
        bar_frame.pack()
        
        for i in range(10):
            row_frame = tk.Frame(bar_frame)
            row_frame.pack(pady=2)
            
            label = tk.Label(row_frame, text=f"{i}:", width=3, font=('Arial', 10))
            label.pack(side=tk.LEFT)
            
            progress = ttk.Progressbar(row_frame, length=300, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5)
            
            value_label = tk.Label(row_frame, text="0%", width=6, font=('Arial', 10))
            value_label.pack(side=tk.LEFT)
            
            self.confidence_bars.append(progress)
            self.confidence_labels.append(value_label)
        
        # Status bar
        self.status_bar = tk.Label(root, text="Ready - Draw a digit", 
                                    bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def paint(self, event):
        """Draw on the canvas and update the grid"""
        # Calculate which cell the mouse is in
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # Draw a 3x3 brush for better drawing experience
            brush_size = 3
            for i in range(-brush_size//2, brush_size//2 + 1):
                for j in range(-brush_size//2, brush_size//2 + 1):
                    nx, ny = x + i, y + j
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        # Add intensity based on distance from center
                        distance = np.sqrt(i**2 + j**2)
                        intensity = 255 * (1 - distance/brush_size)
                        self.drawing_grid[ny, nx] = min(255, self.drawing_grid[ny, nx] + intensity)
                        
                        # Draw on canvas
                        x1 = nx * self.cell_size
                        y1 = ny * self.cell_size
                        x2 = x1 + self.cell_size
                        y2 = y1 + self.cell_size
                        
                        # Color intensity (white to light gray)
                        color_intensity = int(self.drawing_grid[ny, nx])
                        color = f'#{color_intensity:02x}{color_intensity:02x}{color_intensity:02x}'
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
    
    def reset_canvas(self):
        """Clear the canvas and reset the grid"""
        self.drawing_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.canvas.delete("all")
        
        # Redraw blank grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='#333333')
        
        self.prediction_label.config(text="Canvas cleared - Draw a digit", fg='blue')
        self.status_bar.config(text="Canvas reset")
        
        # Reset confidence bars
        for bar, label in zip(self.confidence_bars, self.confidence_labels):
            bar['value'] = 0
            label.config(text="0%")
    
    def preprocess_drawing(self):
        """Convert drawing to format expected by network"""
        # Normalize to [0, 1] range and reshape to 784-dim vector
        normalized = self.drawing_grid / 255.0
        return normalized.flatten()
    
    def predict_digit(self):
        """Run forward pass through the network and display prediction"""
        # Check if canvas is empty
        if np.max(self.drawing_grid) < 10:
            self.prediction_label.config(text="Please draw a digit first!", fg='red')
            self.status_bar.config(text="Error: No digit detected")
            return
        
        # Preprocess drawing
        input_vector = self.preprocess_drawing()
        
        # Get prediction from network
        output = self.net.forward_pass(input_vector)
        
        # Get predicted digit
        predicted_digit = int(np.argmax(output))
        confidence = float(output[predicted_digit]) * 100
        
        # Update prediction label
        self.prediction_label.config(
            text=f"PREDICTION: {predicted_digit} (Confidence: {confidence:.1f}%)", 
            fg='green'
        )
        
        # Update confidence bars
        for i, (bar, label) in enumerate(zip(self.confidence_bars, self.confidence_labels)):
            prob = float(output[i]) * 100
            bar['value'] = prob
            label.config(text=f"{prob:.1f}%")
            
            # Color code the highest confidence
            if i == predicted_digit:
                bar['style'] = 'green.Horizontal.TProgressbar'
            else:
                bar['style'] = 'blue.Horizontal.TProgressbar'
        
        self.status_bar.config(text=f"Predicted digit: {predicted_digit} with {confidence:.1f}% confidence")
        
        # Optionally show the preprocessed image in a separate window
        self.show_preprocessed_image()
    
    def show_preprocessed_image(self):
        """Show how the network sees the drawing"""
        # Create a new window
        top = tk.Toplevel(self.root)
        top.title("Network's View of Your Drawing")
        top.geometry("300x300")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.drawing_grid, cmap='gray', interpolation='nearest')
        ax.set_title("Processed 28x28 Image")
        ax.axis('off')
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack()

def launch_drawing_app(network):
    """Launch the drawing application after training"""
    root = tk.Tk()
    
    # Configure ttk styles for colored progress bars
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
    style.configure("blue.Horizontal.TProgressbar", foreground='blue', background='blue')
    
    app = MNISTDrawingApp(root, network)
    
    # Initialize blank grid
    app.reset_canvas()
    
    root.mainloop()

# Modified main execution code
if __name__ == "__main__":
    # Your existing training code here...
    # ... (load data, train network) ...
    
    # After training is complete, launch the drawing app
    print("\nTraining complete! Launching drawing application...")
    launch_drawing_app(mnist_Net)