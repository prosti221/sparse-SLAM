import yaml
from slam.utils.logger import warning_log

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
                LOG_TAG, f"Property '{prop}' not found for video '{key}'. Using default value: {default}")
            return default

        return self.config['videos'][key][prop]

    def get_global_config_property(self, prop, default=None):
        if prop not in self.config['global']:
            warning_log(
                LOG_TAG, f"Property '{prop}' not found in global config. Using default value: {default}")
            return default

        return self.config['global'][prop]

    def __str__(self):
        ret = ""
        video_tag = self.get_global_config_property('load_video')

        ret += f"\nVideo: {video_tag}"
        ret += f"\nPath: {self.config['videos'][video_tag]['path']}"
        ret += f"\nFeature extractor: {self.get_global_config_property('feature_extractor', default='undefined')}"
        ret += f"\nEnable multiscale features: {self.get_global_config_property('enable_multiscale_features', default='undefined')}"
        ret += f"\nEnable bundle adjustment: {self.get_global_config_property('enable_ba', default='undefined')}"
        ret += f"\nCamera intrinsics:"
        ret += f"\n  Fx: {self.get_video_property(video_tag, 'fx', default='undefined')}"
        ret += f"\n  Fy: {self.get_video_property(video_tag, 'fy', default='undefined')}"
        ret += f"\n  Cx: {self.get_video_property(video_tag, 'cx', default='undefined')}"
        ret += f"\n  Cy: {self.get_video_property(video_tag, 'cy', default='undefined')}"

        return ret
