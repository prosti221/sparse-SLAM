import yaml

class Parser:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.parse()


    def parse(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_video_property(self, key, prop):
        """
        Currently only has video properties
        key: video
        prop: (path, focal_length)
        """
        return self.config['videos'][key][prop]
    

    def __str__(self):
        ret = ""
        for video_name in self.config['videos']:
            ret += f"\nVideo: {video_name}"
            ret += f"\nPath: {self.config['videos'][video_name]['path']}"
            ret += f"\nFocal Length: {self.config['videos'][video_name]['focal_length']}"
            ret += "\n"
        
        return ret

        


