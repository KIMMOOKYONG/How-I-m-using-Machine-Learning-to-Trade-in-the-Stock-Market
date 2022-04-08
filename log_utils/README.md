# logger 사용법

```python

!git clone https://github.com/KIMMOOKYONG/utils.git
mv ./utils/log_utils ./
rm -rf ./utils

# 테스트 코드
# 아래 코드를 로깅할 모듈의 최상단에 포함 시킨다.
import logging
import log_utils.logger_init
logger = logging.getLogger("__main__")

logger.warning("Woohoo I am Logging!")
logger.warning("Woohoo I am Logging!")
logger.warning("Woohoo I am Logging!")
logger.warning("Woohoo I am Logging!")

```


# 라이브러리 의존성 설치 필요
- !pip install -U PyYAML

# Python’s built-in logging tools
- 개발 작업에서 디버깅 전략은 매우 매우 중요하다. 특히 논리적인 오류 잡아야하는 경우와 백엔드 개발의 경우
- discord channel로 에러 메시지 출력 기능

# loggers and handlers 
![](https://miro.medium.com/max/622/1*Hp-v16ZQbei_bAyvXE2KAw.png)

## Loggers
- 모든 logger 파이썬 프로그램 내에서는 global이다.
- logger의 핵심 역활은 LogRecords를 handler로 전송하는 것이다.

```python
# app.py
# logger 인스턴스 생성
# logger 인스터스 이름은 app이다.
# 적용범위는 global
import logging
logger = logging.getLogger("app")



# 아래는 개선된 코드
# 각 파일별로 유닉한 logger를 가질수 있다.???
# app.py
import logging
# logger 인스턴스 생성
# logger 인스터스 이름은 app이다.
# 적용범위는 global
logger = logging.getLogger(__name__)
```

## Handlers(standalone objects)
- 핸들러의 역활은 LogRecords를 어디로 보낼지를 결정한다.
- 핸들러의 종류는 다양

```python
# LogRecord 정보를 ieddit.log 파일로 저장
# 모든 로깅 정보를 기록하므로 로거 파일 용량 이슈가 발생할 여지가 있다.
logger = logging.getLogger(__name__)
fileHandle = logging.FileHandler('ieddit.log')
logger.addHandler(fileHandle)


# setLevel 기능을 이용해서 로깅할 LogRecord를 필터링 할 수 있다.
logger = logging.getLogger(__name__)
fileHandle = logging.FileHandler('ieddit.log')
fileHandle.setLevel(logging.WARNING)
logger.addHandler(fileHandle)

logger.info("This won't show in ieddit.log")
logger.error("This will show.")
```

## Formatters
- Where, what, why, when, and how 등의 부가적인 정보를 생성한다.

```python
# logger 생성
logger = logging.getLogger("__main__")
consoleHandle = logging.StreamHandler()
consoleHandle.setLevel(logging.INFO)

# Setup the formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandle.setFormatter(formatter)
logger.addHandler(consoleHandle)
logger.info("Message")

# 2019-10-25 00:01:42,283 - 모듈명 - INFO - Message

```

## QueueHandler and QueueListener
- 프로그램에서 로깅 메시지를 출력하면, 로깅 처리가 완료 될 때까지 프로그램 기다려야하는데
- 별도의 쓰레스로 실행되는 QueueHandler, QueueListener를 사용하면 기다리지 않아도 된다.
- QueueListener - > QueueHandler

```python
# run.py
import logging_util
import logging
from queue import Queue

# Initialize Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
fileHandle = logging.FileHandler('ieddit.log')
queueHandle = logging_util.QueueListenerHandler(
                                     queue=Queue(-1), 
                                     handlers=[fileHandle])
logger.warning("Ack!")

# 아래는 처리 알고리즘 설명
Warning Log Record Created
   |
   V
Log record added to QueueHandler on separate thread
 -> Main program is no longer blocked
   |
   V
QueueListener picks up the log and hands it to the other handlers for processing
   |
   V
Other handlers format the message and output appropriately

```

## Custom Discord Handler
```python
class DiscordHandler(logging.Handler):
    """
    Custom handler to send certain logs to Discord
    """
    def __init__(self):
        logging.Handler.__init__(self)
        self.discordWebhook = DiscordWebhook(url=config.DISCORD_URL)

    def emit(self, record):
        """
        sends a message to discord
        """
        desc = [
            record.message,
            record.exc_info,
            str(record.funcName) + " : " + str(record.lineno),
            record.stack_info
            ]

        filteredDesc = [record for record in desc if record != None]

        embed = DiscordEmbed(
            title=record.levelname,
            description="\n".join(filteredDesc),
            color=16711680)
        self.discordWebhook.add_embed(embed)
        self.discordWebhook.execute()
```

## Using a File to Initialize Loggers
### logger 환경설정
```yaml
# Logger Config.yaml
version: 1
objects:
  queue:
    class: queue.Queue
    maxsize: 1000
formatters:
  simple:
    format: '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
handlers:
  discord:
    class: utilities.log_utils.logger_util.DiscordHandler
    level: ERROR
    formatter: simple
  queue_listener:
    class: utilities.log_utils.logger_util.QueueListenerHandler
    handlers:
      - cfg://handlers.console
      - cfg://handlers.file
      - cfg://handlers.discord
    queue: cfg://objects.queue
loggers:
  __main__:
    level: WARNING
    handlers:
      - queue_listener
    propagate: false
  ieddit:
    level: WARNING
    handlers:
      - queue_listener
    propagate: false
```

### 활용예시
```python
# logger_init.py
import os
import config
# Setup For Logging Init
import yaml
import logging
import utilities.log_utils.logger_util
# Pull in Logging Config
with open('logger_config.yaml', 'r') as stream:
  try:
    logging_config = yaml.load(stream, Loader=yaml.SafeLoader)
  except yaml.YAMLError as exc:
    print("Error Loading Logger Config")
    pass
# Load Logging configs
logging.config.dictConfig(logging_config)
```

```python
# run.py
import logging
import logger_init # Only call at the top file in 
import logging
logger = logging.getLogger("__main__")
logger.warning("Woohoo I am Logging!")
```


# Logging Utils

These logging utilities create default loggers for all of the files. The configuration is stored on a static config file logger_config.yaml. 

## How Loggers work

Loggers use objects to send logs places. Each module should have its own logger with a name. Data flows like this in the application;

Logger -> QueueListener -> Other handers (FileHandler, Console) 

Each level works with filtering by the Log Levels below:

Level | Numberic Value
----- | --------------
CRITICAL | 50
ERROR | 40
WARNING | 30
INFO | 20
DEBUG | 10
NOTSET | 0

It works from the bottom up. If the logging level that handler/logger will only catch at that level and above. 

## Updating Logging Yaml

### Objects
```yaml
objects:
  queue:
    class: queue.Queue
    maxsize: 1000
```
 
 Objects represent arbitrary python class objects that are needed. Update maxsize if needed.


### Formatters
```yaml
formatters:
  simple:
    format: '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
```

Formatters specify the format that logs should be in. Formatters are childeren of handlers so if a different fomatter is need a seperate handler may be needed. 


### Handlers
```yaml
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: simple
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    level: WARNING
    filename: 'ieddit.log'
    formatter: simple
  queue_listener:
    class: utilities.log_utils.logger_util.QueueListenerHandler
    handlers:
      - cfg://handlers.console
      - cfg://handlers.file
    queue: cfg://objects.queue
```

Handlers define where the logs should go when log statements are hit. Queue handler here defines that after items hit it they should go into console and then file.

1. Stream handler:
    * Sends output to the console. It is set to allow everything debug and higher in priority. 
2. File handler:
    * Sends output to specified file. It is set to allow everything warning and higher in priority. This is to prevent logs from growing too large. 
3. Queue Listener:
    * This is a very special handler. It lives on a seperate thread from the application so log statements do not block the application flow. All loggers should point here. 

### Loggers
```yaml
loggers:
    blueprints.admin:
    level: WARNING
    handlers: 
      - queue_listener
    propagate: false
```

Logger objects are the individual instances of the logging class. Each file should have its own logger defined by its name. Above is what the logger looks like for admin.py which is in the blueprints module. Its data is directed at the queue-listener which then spits the data out to the 2 other handlers. Propogate means do we want this error bubbling up to other loggers, which we do not want. 

## Using Loggers in files

```python
# Admin.py
# Logging Initialization
import logging 
logger = logging.getLogger(__name__)
```

Here the default log level for production is WARNING and above. If we wanted to do different we could manually set the level in a file like so:

```python
# Admin.py
# Logging Initialization
import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

This will override what is set in the config. If you set this to debug however it will only be caught by the console becuase the FileHandler filters anything out below Warning.
