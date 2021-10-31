import datetime
import watchtower, logging
from abc import ABCMeta, abstractmethod
import traceback


class LoggerFactory:
    @classmethod
    def get_logger(
        cls, logger_type: str, logger_name: str, boto3_session=None
    ):
        if logger_type == "print":
            return PrintLogger(logger_name=logger_name)
        elif logger_type == "cloudwatch":
            return CloudWatchLogger(
                boto3_session=boto3_session, logger_name=logger_name
            )
        else:
            raise NotImplementedError(
                "{} logger type is not implemented!".format(logger_type)
            )


class AbstractLogger(object):
    __metadata__ = ABCMeta

    @abstractmethod
    def info(self, message: str):
        pass

    @abstractmethod
    def error(self, message: str):
        pass

    @abstractmethod
    def error_detail(self, err):
        pass

    def _log_message(self, message) -> str:
        return "[%s] %s" % (self._current_datetime(), message)

    def _current_datetime(self):
        return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def _create_error_detail_txt(self, err: Exception) -> str:
        err_txt = "{}".format(err)
        err_detail_txt = "{}".format(traceback.format_exc())
        return err_txt, err_detail_txt


class CloudWatchLogger(AbstractLogger):
    def __init__(self, boto3_session, logger_name: str) -> None:
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(logger_name)
        self._logger.addHandler(
            watchtower.CloudWatchLogHandler(boto3_session=boto3_session)
        )

    def info(self, message):
        self._logger.info(self._log_message(message))

    def error(self, message):
        self._logger.error(self._log_message(message))

    def error_detail(self, err):
        err_txt, err_detail_txt = self._create_error_detail_txt(err)
        self.error(err_txt)
        self._logger.error(self._log_message(err_detail_txt))


class PrintLogger(AbstractLogger):
    def __init__(self, logger_name: str) -> None:
        self._logger_name = logger_name

    def info(self, message):
        print(
            "INFO: " + self._logger_name + " : " + self._log_message(message)
        )

    def error(self, message):
        print(
            "ERROR: " + self._logger_name + " : " + self._log_message(message)
        )

    def error_detail(self, err: Exception):
        err_txt, err_detail_txt = self._create_error_detail_txt(err)
        self.error(err_txt)
        print(
            "ERROR Detail: "
            + self._logger_name
            + " : "
            + self._log_message(err_detail_txt)
        )
