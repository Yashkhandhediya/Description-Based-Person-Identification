def Singleton(klass):
    if not klass._instance:
        klass._instance = klass()
    return klass._instance


class Data_bridge:
    """
    This class is data bridge between all the class defined in our module.
    This class contains all types of data that's to be shared between multiple classes.
    """
    _instance = None
    def __init__(self):
        self.methode_chosen_by_radio_butten = ''
        self.methode_chosen_for_tracking=''
        self.start_process_manager = False
        self.save_data_in_video_format = False
        self.save_data_in_image_format = True
        self.selected_video_file_path = ''
        self.selected_directory_path = ''
        pass