import os

# Get Application Config
# import config

# Setup For Logging Init
import yaml
import logging
import logging.config
# import utilities.log_utils.logger_util

# Pull in Logging Config
# path = os.path.join(os.getcwd(), 'utilities', 'log_utils', 'logger_config.yaml')
path = os.path.join(os.getcwd(), 'log_utils', 'logger_config.yaml')
with open(path, 'r') as stream:
    try: # pip install -U PyYAML
      logging_config = yaml.load(stream, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
      print("Error Loading Logger Config")
      pass

# Load Logging configs
logging.config.dictConfig(logging_config)

# Initialize Log Levels
# 로깅 레벨 초기값 설정
# log_level = logging.WARNING

# Check For Debug Flag
# if config.DEBUG:
#  log_level = logging.DEBUG

# Set the logging level for all loggers in scope 
# This level can be overwritten by the following in a file
#   logger = logging.getlogger(__name__)
#   logger.setLevel(logging.INFO)

# 전체 로그 레벨을 단일 값으로 설정하고자 하는 경우 아래 코드를 적용한다.
# log_level = logging.WARNING
# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# for log in loggers:
#   log.setLevel(log_level)
