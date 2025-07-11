import yaml
from logger import warning_log

LOG_TAG = 'Parser'


class Parser:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.parse()

    def parse(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_video_property(self, key, prop, default=None):
        """
        Currently only has video properties
        key: video
        prop: (path, focal_length)
        """
        if prop not in self.config['videos'][key]:
            warning_log(
                LOG_TAG, f"Prop '{prop}' not found for video '{key}'. Using default value: {default}")
            print(f"Prop '{prop}' not found in config. Using default value.")
            return default

        return self.config['videos'][key][prop]

    def __str__(self):
        ret = ""
        for video_name in self.config['videos']:
            ret += f"\nVideo: {video_name}"
            ret += f"\nPath: {self.config['videos'][video_name]['path']}"
            ret += f"\nCamera intrinsics:"
            ret += f"\n  Fx: {self.get_video_property(video_name, 'fx', default='undefined')}"
            ret += f"\n  Fy: {self.get_video_property(video_name, 'fy', default='undefined')}"
            ret += f"\n  Cx: {self.get_video_property(video_name, 'cx', default='undefined')}"
            ret += f"\n  Cy: {self.get_video_property(video_name, 'cy', default='undefined')}"
            ret += "\n"
        return ret
