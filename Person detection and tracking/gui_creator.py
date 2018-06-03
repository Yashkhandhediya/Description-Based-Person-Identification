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
        self.root.title("Description Based Person Identification")
        self.content = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        self.chosen_method = StringVar()
        self.chosen_track=StringVar()

    def defining_labels(self):
        # Title label
        self.title_label = ttk.Label(self.content, text="Video Analysis Screen")
        self.title_label.config(font=("Courier", 30))

        # Video select label
        self.video_select_label = ttk.Label(self.content, text="Path of video selected by you will be displayed here")

        # directory select label
        self.dir_select_label = ttk.Label(self.content, text="Path of target folder selected by you will be displayed here")

        #methods label
        # directory select label
        self.methods_label = ttk.Label(self.content, text="Methods")
        self.methods_label.config(font=("Courier", 24))

        self.tracks_label = ttk.Label(self.content, text="Tracking algorithm")
        self.tracks_label.config(font=("Bold", 16))

    def defining_buttons(self):
        # video select button
        self.video_select_button = ttk.Button(self.content, text="Video select", command=self.select_video_file)

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
        self.Mil = ttk.Radiobutton(self.content, text='MIL', variable=self.chosen_track, value='MIL')
        self.Kcf = ttk.Radiobutton(self.content, text='KCF', variable=self.chosen_track, value='KCF')
        self.Tld = ttk.Radiobutton(self.content, text='TLD', variable=self.chosen_track, value='TLD')
        self.Boosting = ttk.Radiobutton(self.content, text='BOOSTING', variable=self.chosen_track, value='BOOSTING')
        self.Medianflow = ttk.Radiobutton(self.content, text='MEDIANFLOW', variable=self.chosen_track, value='MEDIANFLOW')

        self.raw_video = ttk.Radiobutton(self.content, text='Raw video', variable=self.chosen_method, value='raw_video')
        self.yolo_pd = ttk.Radiobutton(self.content, text='YOLO person detection', variable=self.chosen_method, value='yolo_pd')

    def process_video_method(self):
        print("chosen method = ", self.chosen_method.get())
        self.data_bridge.methode_chosen_by_radio_butten = self.chosen_method.get()
        self.data_bridge.methode_chosen_for_tracking=self.chosen_track.get()
        print("chosen tracking = ", self.data_bridge.methode_chosen_for_tracking)
        self.data_bridge.start_process_manager = True
        pass

    def stop_processing_video(self):
        print("process stops ")
        self.data_bridge.start_process_manager = False
        pass


    def defining_geometry_grid(self):
        self.content.grid(column=0, row=0, sticky=(N, S, E, W))
        self.title_label.grid(column=1, row=0, columnspan=2, sticky=(N, W), padx=5)
        self.video_select_label.grid(column=3, row=2, columnspan=2, sticky=(N, W))
        self.dir_select_label.grid(column=3, row=3, columnspan=2, sticky=(N, W))
        self.raw_video.grid(column=0, row=6)
        self.yolo_pd.grid(column=0, row=7)
        self.Kcf.grid(column=3, row=5,sticky=(W))
        self.Mil.grid(column=3, row=6,sticky=(W))
        self.Tld.grid(column=3, row=7,sticky=(W))
        self.Boosting.grid(column=3, row=8,sticky=(W))
        self.Medianflow.grid(column=3, row=9,sticky=(W))
        self.process_video_button.grid(column=0, row=8)
        self.stop_video_processing_button.grid(column=0, row=9)
        self.video_select_button.grid(column=0, row=2)
        self.target_folder_button.grid(column=0, row=3)
        self.methods_label.grid(column = 0, row = 5)
        self.tracks_label.grid(column=3, row=4)


    def defining_whole_ui(self):
        self.defining_labels()
        self.defining_buttons()
        self.define_radio_buttons_for_method_select()
        self.defining_geometry_grid()

    def update(self):
        print ("Here")
        self.root.mainloop()



# g = GUI_Creator_temp()
# g.defining_whole_ui()
# g.run_loop()
