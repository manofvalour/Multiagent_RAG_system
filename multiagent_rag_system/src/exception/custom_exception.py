import traceback
from typing import Optional, cast
import sys
from ..logger.logger import GLOBAL_LOGGER as logger

class MulitagentragException(Exception):
    def __init__(self, error_message, error_details: Optional[object]=None):
        if isinstance(error_message, BaseException):
            norm_arg = str(error_message)

        else:
            norm_arg = str(error_message)

        exc_type = exc_value = exc_tb = None
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        
        else:
            if hasattr(error_details, 'exc_info'):
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__

            else:
                exc_type, exc_value, exc_tb = sys.exc_info()
        
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_arg

        if exc_type and exc_tb:
            self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""
        
        super().__init__(self.__str__())

    def __str__(self):
        base = f"Eror in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        
        return base

    def __repr__(self):
        return f"MultiagentragException(file={self.file_name}, line={self.lineno}, message = {self.error_message})"
    
if __name__ =="__main__":
    try:
        a = 1/0
    except Exception as e:
        logger.error("Division by zero error")
        raise MulitagentragException("Division failed", e ) from e