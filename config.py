import json


class Config:
    def __init__(self, config_file):
        """
        初始化配置类，加载 JSON 配置文件
        :param config_file: JSON 文件路径
        """
        with open(config_file, 'r') as f:
            self.params = json.load(f)

    def __getattr__(self, name):
        """
        通过属性名称访问配置项
        :param name: 配置项的名称
        :return: 配置项的值
        """
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"Configuration parameter '{name}' not found.")

    def update(self, **kwargs):
        """
        更新配置项
        :param kwargs: 要更新的配置项及其值
        """
        self.params.update(kwargs)

    def get(self, name, default=None):
        """
        获取配置项的值，如果配置项不存在则返回默认值
        :param name: 配置项的名称
        :param default: 默认值
        :return: 配置项的值或默认值
        """
        return self.params.get(name, default)