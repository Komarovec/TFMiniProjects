
import tkinter as tk
from tkinter import filedialog
import kivy

kivy.require('1.0.6')
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout


class Canvas(GridLayout):
    def __init__(self, **kwargs):
        super(Canvas, self).__init__(**kwargs)
        self.image = Image(source='image.jpg')
        self.btn = Button(text="Change image")
        self.btn.bind(on_press=self.btnPressed)
        self.cols = 2
        self.add_widget(self.btn)
        self.add_widget(self.image)

    def btnPressed(self, touch):
        tk.Tk().withdraw()
        in_path = filedialog.askopenfilename()
        if(in_path != ''):
            self.image.source = in_path

class PicturesApp(App):
    def build(self):
        return Canvas()

if __name__ == '__main__':
    PicturesApp().run()