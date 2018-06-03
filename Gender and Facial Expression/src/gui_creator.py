from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from data_bridge import *



class GUI_Creator_temp:
    """
    This class creates gui on passed parent frame object.
    """

    def __init__(self):
        self.data_bridge = Singleton(Data_bridge)
        self.root = Tk()
        self.root.title("Gender and emotion classifier")
        self.content = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        self.chosen_method = StringVar()
        self.chosen_process=StringVar()

    def defining_labels(self):
        # Title label
        self.title_label = ttk.Label(self.content, text="Video Analysis Screen")
        self.title_label.config(font=("Courier", 30))

        # Video select label
        self.video_select_label = ttk.Label(self.content, text="Path of file selected by you will be displayed here")

        # directory select label
        self.dir_select_label = ttk.Label(self.content, text="Path of target folder selected by you will be displayed here")

        #methods label
        # directory select label
        self.methods_label = ttk.Label(self.content, text="Methods")
        self.methods_label.config(font=("Courier", 24))

        self.process_label = ttk.Label(self.content, text="Processing source from")
        self.process_label.config(font=("Bold", 16))


    def defining_buttons(self):
        # video select button
        self.video_select_button = ttk.Button(self.content, text="File select", command=self.select_video_file)
        # directory select button
        self.target_folder_button = ttk.Button(self.content, text="Target folder select", command=self.select_directory)
        # process video button
        self.process_video_button = ttk.Button(self.content, text="Process video", command=self.process_video_method)
        # stop video processing video button
        self.stop_video_processing_button = ttk.Button(self.content, text="Stop video processing", command=self.stop_processing_video)

    def select_video_file(self):
        self.selected_video_file_path = filedialog.askopenfilename()
        self.video_select_label["text"] = self.selected_video_file_path
        self.data_bridge.selected_video_file_path = self.selected_video_file_path

    def select_directory(self):
        self.selected_directory_path = filedialog.askdirectory()
        self.dir_select_label["text"] = self.selected_directory_path
        self.data_bridge.selected_directory_path = self.selected_directory_path

    def define_radio_buttons_for_method_select(self):
        self.from_image = ttk.Radiobutton(self.content, text='Image', variable=self.chosen_process, value='img')
        self.from_video = ttk.Radiobutton(self.content, text='Video', variable=self.chosen_process, value='vid')
        self.from_webcam = ttk.Radiobutton(self.content, text='Webcam', variable=self.chosen_process, value='web')
        self.raw_file = ttk.Radiobutton(self.content, text='Raw file', variable=self.chosen_method, value='raw_file')
        self.face_det = ttk.Radiobutton(self.content, text='Feature detection', variable=self.chosen_method, value='face_det')

    def process_video_method(self):
        print("chosen method = ", self.chosen_method.get())
        self.data_bridge.methode_chosen_by_radio_butten = self.chosen_method.get()
        self.data_bridge.processing_chosen_by_radio_butten=self.chosen_process.get()
        print("Chosen source file = ", self.data_bridge.processing_chosen_by_radio_butten)
        self.data_bridge.start_process_manager = True
        pass

    def stop_processing_video(self):
        print("Process stops ")
        self.data_bridge.start_process_manager = False
        pass


    def defining_geometry_grid(self):
        self.content.grid(column=0, row=0, sticky=(N, S, E, W))
        self.title_label.grid(column=1, row=0, columnspan=2, sticky=(N, W), padx=5)
        self.video_select_label.grid(column=3, row=2, columnspan=2, sticky=(N, W))
        self.dir_select_label.grid(column=3, row=3, columnspan=2, sticky=(N, W))
        self.raw_file.grid(column=0, row=6)
        self.face_det.grid(column=0, row=7)
        self.from_image.grid(column=3, row=5,sticky=(W))
        self.from_video.grid(column=3, row=6,sticky=(W))
        self.from_webcam.grid(column=3, row=7,sticky=(W))
        self.process_video_button.grid(column=0, row=8)
        self.stop_video_processing_button.grid(column=0, row=9)
        self.video_select_button.grid(column=0, row=2)
        self.target_folder_button.grid(column=0, row=3)
        self.methods_label.grid(column = 0, row = 5)
        self.process_label.grid(column=3, row=4)


    def defining_whole_ui(self):
        self.defining_labels()
        self.defining_buttons()
        self.define_radio_buttons_for_method_select()
        self.defining_geometry_grid()

    def update(self):
        print ("Here")
        self.root.mainloop()
